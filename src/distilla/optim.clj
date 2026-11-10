(ns distilla.optim
  "AdamW over a memory module's parameters, gradient-norm clipping, and the
  learning-rate schedule of train.py (linear warmup from 1e-8 of the peak,
  then cosine decay to 10% of it).

  The per-element loops run in native/distilla_ops when it is loaded (`jolt
  native`), otherwise in Clojure, which is fine for small modules and tests."
  (:require [distilla.blas :as blas]
            [distilla.memory :as memory]
            [distilla.safetensors :as st]
            [jolt.ffi :as ffi]))

(ffi/defcfn dops-adamw "dops_adamw" [:pointer :pointer :pointer :pointer :int64 :float :float :float :float :float :float :float] :void)
(ffi/defcfn dops-sumsq "dops_sumsq" [:pointer :int64] :double)
(ffi/defcfn dops-scale "dops_scale" [:pointer :int64 :float] :void)

(def warmup-start-factor 1e-8)
(def min-lr-ratio 0.1)

(defn native? [] (some? (ffi/find-symbol "dops_adamw")))

;; --- schedule ---------------------------------------------------------------------------------

(defn lr-schedule
  "The learning rate of optimizer step i (0-based), as torch's
  SequentialLR(LinearLR(1e-8 -> 1, warmup), CosineAnnealingLR(eta_min =
  0.1 peak)) computes it. Without warmup steps the decay starts at the peak."
  [{:keys [peak total-steps warmup-ratio min-ratio] :or {min-ratio min-lr-ratio warmup-ratio 0.0}}]
  (let [warmup (int (* warmup-ratio total-steps))
        eta-min (* peak min-ratio)
        cosine (fn [i t] (+ eta-min (* (- peak eta-min) 0.5 (+ 1.0 (Math/cos (/ (* Math/PI i) t))))))]
    (fn [i]
      (cond
        (zero? warmup) (cosine i (max total-steps 1))
        (< i warmup) (* peak (+ warmup-start-factor (* (- 1.0 warmup-start-factor) (/ (double i) warmup))))
        :else (cosine (- i warmup) (max (- total-steps warmup) 1))))))

(defn constant-schedule [lr] (fn [_] lr))

;; --- gradients ----------------------------------------------------------------------------------

(defn grad-norm
  "The global L2 norm of the module's gradients."
  [module]
  (let [grads (memory/ensure-grads! module)]
    (Math/sqrt
     (reduce + 0.0
             (for [[name p] grads]
               (let [n (get-in module [:params name :n])]
                 (if (native?)
                   (dops-sumsq p n)
                   (let [^floats g (blas/foreign->floats p n)]
                     (loop [i 0 s 0.0] (if (< i n) (recur (inc i) (+ s (* (aget g i) (aget g i)))) s))))))))))

(defn scale-grads! [module factor]
  (doseq [[name p] (memory/ensure-grads! module)]
    (let [n (get-in module [:params name :n])]
      (if (native?)
        (dops-scale p n (float factor))
        (let [^floats g (blas/foreign->floats p n)]
          (dotimes [i n] (aset g i (float (* factor (aget g i)))))
          (ffi/write-array p :float g))))))

(defn clip-grad-norm!
  "torch.nn.utils.clip_grad_norm_: scale the gradients so their global norm
  is at most max-norm. Returns the norm before clipping."
  [module max-norm]
  (let [norm (grad-norm module)
        coef (/ max-norm (+ norm 1e-6))]
    (when (< coef 1.0) (scale-grads! module coef))
    norm))

;; --- AdamW ----------------------------------------------------------------------------------------

(defn adamw
  "AdamW state for a module (torch defaults: betas 0.9/0.999, eps 1e-8)."
  [module {:keys [weight-decay beta1 beta2 eps] :or {weight-decay 0.01 beta1 0.9 beta2 0.999 eps 1e-8}}]
  (let [arena (:arena module)
        zeros (fn [] (into {} (for [[name {:keys [n]}] (:params module)]
                                (let [p (blas/alloc-floats arena n)]
                                  (ffi/write-array p :float (float-array n))
                                  [name p]))))]
    {:module module :m (zeros) :v (zeros) :step (atom 0)
     :weight-decay weight-decay :beta1 beta1 :beta2 beta2 :eps eps}))

(defn- adamw-clj [p g m v n lr b1 b2 eps wd bc1 bc2]
  (let [^floats pa (blas/foreign->floats p n)
        ^floats ga (blas/foreign->floats g n)
        ^floats ma (blas/foreign->floats m n)
        ^floats va (blas/foreign->floats v n)
        decay (- 1.0 (* lr wd))]
    (dotimes [i n]
      (let [gi (aget ga i)
            mi (+ (* b1 (aget ma i)) (* (- 1.0 b1) gi))
            vi (+ (* b2 (aget va i)) (* (- 1.0 b2) gi gi))]
        (aset ma i (float mi))
        (aset va i (float vi))
        (aset pa i (float (- (* (aget pa i) decay)
                             (* lr (/ (/ mi bc1) (+ (Math/sqrt (/ vi bc2)) eps))))))))
    (ffi/write-array p :float pa)
    (ffi/write-array m :float ma)
    (ffi/write-array v :float va)))

(defn step!
  "One AdamW update of every parameter from its gradient, at learning rate lr."
  [{:keys [module m v step weight-decay beta1 beta2 eps]} lr]
  (let [t (swap! step inc)
        bc1 (- 1.0 (Math/pow beta1 t))
        bc2 (- 1.0 (Math/pow beta2 t))
        grads (memory/ensure-grads! module)]
    (doseq [[name {:keys [ptr n]}] (:params module)]
      (if (native?)
        (dops-adamw ptr (grads name) (m name) (v name) n (float lr) (float beta1) (float beta2) (float eps)
                    (float weight-decay) (float bc1) (float bc2))
        (adamw-clj ptr (grads name) (m name) (v name) n lr beta1 beta2 eps weight-decay bc1 bc2)))
    t))

(defn save-state
  "Write the optimizer moments to path (safetensors, step in the metadata)."
  [{:keys [module m v step]} path]
  (st/save-file (into {} (for [[name {:keys [shape n]}] (:params module)
                               [kind ptrs] [["m" m] ["v" v]]]
                           [(str kind "." name) {:dtype "F32" :shape shape :data (blas/foreign->floats (ptrs name) n)}]))
                path {"step" (str @step)}))

(defn load-state!
  "Restore the optimizer moments and step written by save-state."
  [{:keys [module m v step]} path]
  (let [s (st/open path)]
    (doseq [[name {:keys [n]}] (:params module)
            [kind ptrs] [["m" m] ["v" v]]]
      (ffi/write-array (ptrs name) :float (:data (st/read-tensor s (str kind "." name)))))
    (reset! step (parse-long (get (:metadata s) "step" "0")))))
