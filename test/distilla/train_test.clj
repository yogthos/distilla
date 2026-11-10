(ns distilla.train-test
  "Training: the memory module's backward pass against PyTorch autograd, the
  optimizer and schedule, recipes, and the training loop with a fake decoder."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [distilla.data :as data]
            [distilla.helpers :as h]
            [distilla.memory :as memory]
            [distilla.optim :as optim]
            [distilla.safetensors :as st]
            [distilla.tensor :as t]
            [distilla.train :as train]
            [distilla.util :as util]))

;; --- backward pass ------------------------------------------------------------------------------

(deftest backward-matches-pytorch-autograd
  (let [{:strs [docs mask question]} (get (util/read-json "test/fixtures/memory/expected.json") "inputs")
        docs (mapv t/from-rows docs)
        questions (mapv #(float-array (map float %)) question)]
    (doseq [name ["default" "retrieval_self1" "self2_cross1"]]
      (testing name
        (with-open [m (memory/from-pretrained (str "test/fixtures/grads/" name))]
          (let [g (st/load-file (str "test/fixtures/grads/" name "/grads.safetensors"))
                ^floats w (:data (g "weights"))
                per (* (memory/num-memories m) 12)]
            (memory/zero-grads! m)
            (dotimes [b 2]
              (let [d (nth docs b) rm (nth mask b)
                    {:keys [output tape]} (memory/forward-tape m (memory/valid-docs d rm) (nth questions b)
                                                               {:slot-docs d :slot-mask rm})]
                (is (h/approx= (vec (:data output))
                               (subvec (vec (:data (g "output"))) (* b per) (* (inc b) per)) 1e-4))
                (memory/backward! m tape (t/matrix (:rows output) (:cols output)
                                                   (java.util.Arrays/copyOfRange w (int (* b per)) (int (* (inc b) per)))))))
            (doseq [p (memory/param-names m)]
              (is (h/approx= (vec (memory/grad-floats m p)) (vec (:data (g (str "grad." p)))) 2e-3) p))))))))

(deftest dropout-only-in-training
  (let [cfg (memory/config {:embedding-dim 16 :output-dim 8 :num-memories 4 :num-heads 2 :dropout 0.5})]
    (with-open [m (memory/init-module cfg 3)]
      (let [docs (t/matrix 5 16 (h/normals 1 80))
            q (h/normals 2 16)
            eval-out (vec (:data (memory/forward m docs q)))
            train-out (fn [seed] (vec (:data (:output (memory/forward-tape m docs q {:train? true :rng (memory/rng seed)})))))]
        (is (= eval-out (vec (:data (memory/forward m docs q)))))
        (is (not= eval-out (train-out 1)))
        (is (= (train-out 1) (train-out 1)))
        (is (not= (train-out 1) (train-out 2)))))))

(deftest initialisation
  (let [cfg (memory/config {:embedding-dim 32 :output-dim 16 :num-memories 4 :num-heads 4})]
    (with-open [m (memory/init-module cfg 7)]
      (is (= (memory/expected-keys cfg) (set (memory/param-names m))))
      (is (every? #(= 1.0 (double %)) (memory/param-floats m "doc_norm.weight")))
      (is (every? zero? (memory/param-floats m "query_cond_attn.in_proj_bias")))
      (let [w (memory/param-floats m "ffn_layers.0.0.weight")
            bound (/ 1.0 (Math/sqrt 32))]
        (is (every? #(<= (Math/abs (double %)) bound) w))
        (is (> (count (distinct (seq w))) 100)))
      (testing "a warm start copies what matches and keeps the rest"
        (with-open [other (memory/init-module (assoc cfg :num-cross-attn-layers 1) 8)]
          (let [{:keys [loaded missing unexpected]} (memory/load-partial! m (memory/state-dict other))]
            (is (pos? loaded))
            (is (some #(str/starts-with? % "cross_attn_layers.1.") missing))
            (is (empty? unexpected))
            (is (= (vec (memory/param-floats other "layer_norm.bias")) (vec (memory/param-floats m "layer_norm.bias"))))))))))

;; --- optimizer ------------------------------------------------------------------------------------

(def steps 100)
(def peak 1e-4)

(deftest learning-rate-schedule
  (doseq [warmup [0.0 0.1]]
    (let [f (optim/lr-schedule {:peak peak :total-steps steps :warmup-ratio warmup})
          rates (mapv f (range steps))]
      (is (< (Math/abs (- peak (nth rates (int (* warmup steps))))) 1e-12))
      (is (<= (apply max rates) (* peak (+ 1 1e-9))))
      (is (< (Math/abs (- (peek rates) (* peak optim/min-lr-ratio))) (* 0.01 peak optim/min-lr-ratio)))))
  (testing "a warmup over every step rises throughout"
    (let [rates (mapv (optim/lr-schedule {:peak peak :total-steps steps :warmup-ratio 1.0}) (range steps))]
      (is (every? true? (map < rates (rest rates)))))))

(defn- tiny-module []
  (memory/init-module {:embedding-dim 8 :output-dim 4 :num-memories 2 :num-heads 2 :dropout 0.0} 1))

(defn- set-grads! [m f]
  (doseq [name (memory/param-names m)]
    (let [n (get-in m [:params name :n])]
      (jolt.ffi/write-array (get (memory/ensure-grads! m) name) :float (float-array (map f (range n)))))))

(deftest adamw-matches-torch-formula
  (with-open [m (tiny-module)]
    (let [opt (optim/adamw m {:weight-decay 0.01})
          name "layer_norm.bias"
          p0 (vec (memory/param-floats m name))]
      (set-grads! m (fn [i] (* 0.1 (inc i))))
      (optim/step! opt 1e-2)
      ;; first step: m = 0.1 g, v = 0.001 g^2, so m_hat / sqrt(v_hat) = sign(g)
      (is (h/approx= (vec (memory/param-floats m name))
                     (map-indexed (fn [i p] (- (* p (- 1 1e-4)) (* 1e-2 (/ 1.0 (+ 1.0 (/ 1e-8 (* 0.1 (inc i)))))))) p0)
                     1e-5))
      (is (= 1 @(:step opt))))))

(deftest gradient-clipping
  (with-open [m (tiny-module)]
    (set-grads! m (constantly 1.0))
    (let [n (memory/parameter-count m)
          norm (optim/clip-grad-norm! m 1.0)]
      (is (< (Math/abs (- norm (Math/sqrt n))) 1e-6))
      (is (< (Math/abs (- 1.0 (optim/grad-norm m))) 1e-4)))))

(deftest optimizer-state-round-trip
  (h/with-temp-dir [dir]
    (with-open [m (tiny-module)]
      (let [opt (optim/adamw m {})
            path (str (fs/path dir "opt.safetensors"))]
        (set-grads! m (constantly 0.5))
        (optim/step! opt 1e-3)
        (optim/save-state opt path)
        (let [fresh (optim/adamw m {})]
          (optim/load-state! fresh path)
          (is (= 1 @(:step fresh)))
          (is (= (vec (jolt.ffi/read-array ((:m opt) "layer_norm.bias") :float 8))
                 (vec (jolt.ffi/read-array ((:m fresh) "layer_norm.bias") :float 8)))))))))

;; --- sequences and the loop ----------------------------------------------------------------------------

(deftest sequence-inputs
  (let [{:keys [tokens mem-index targets]} (train/sequence-inputs [5 9 9 6 7 8] [-100 -100 -100 -100 7 8] 9)]
    (is (= [5 0 0 6 7 8] tokens))
    (is (= [-1 0 1 -1 -1 -1] mem-index))
    (is (= [-1 -1 -1 7 8 -1] targets))))

(defn- fake-decoder
  "A stand-in decoder whose loss is the squared distance of the memory tokens
  from a fixed target (2 scored positions): the loop must drive it down."
  [target]
  (fn [{:keys [memory targets]} grad?]
    (let [^floats d (:data memory)
          n (alength d)
          diff (float-array (map - d (take n (cycle target))))]
      {:loss (/ (reduce + 0.0 (map * diff diff)) n)
       :n-targets (count (filter #(>= % 0) targets))
       :grad (when grad? (t/matrix (:rows memory) (:cols memory) (float-array (map #(* 2.0 (/ % n)) diff))))})))

(defn- examples []
  (for [[i row] (map-indexed vector (h/make-rows))]
    {"question" (row "question") "answer" (row "answer")
     :doc-embeds (t/matrix 3 8 (h/normals i 24)) :question-embed (h/normals (+ 10 i) 8)}))

(deftest training-reduces-the-loss
  (with-open [m (tiny-module)]
    (let [tr (train/trainer {:memory m :decoder (fake-decoder [0.5 -0.25 1.0]) :tokenizer (h/word-tokenizer)
                             :learning-rate 3e-2 :weight-decay 0.0})
          results (train/learn! tr nil (vec (examples)) {:steps 60 :batch-size 2})]
      (is (< (:loss (peek results)) (* 0.2 (:loss (first results)))))
      (is (= 60 (:step (peek results))))
      (testing "snapshot and restore"
        (let [snap (train/snapshot tr)
              before (vec (memory/param-floats m "memory_projection.bias"))]
          (train/learn! tr nil (vec (examples)) {:steps 3})
          (is (not= before (vec (memory/param-floats m "memory_projection.bias"))))
          (train/restore! tr snap)
          (is (= before (vec (memory/param-floats m "memory_projection.bias"))))
          (is (= 60 @(get-in tr [:optimizer :step]))))))))

(deftest batch-loss-weights-tokens
  (with-open [m (tiny-module)]
    (let [tr (train/trainer {:memory m :decoder (fake-decoder [0.0]) :tokenizer (h/word-tokenizer)})
          batch (train/collate tr (vec (take 2 (examples))) {})]
      (is (= 2 (count (:labels batch))))
      (is (pos? (train/eval-loss tr [batch]))))))

(deftest fit-writes-checkpoints-and-resumes
  (h/with-temp-dir [root]
    (h/data-repo root (h/fake-encoder))
    (let [train-set (data/load-split "toy" "train" root)
          stores {["toy" "train"] (distilla.store/open-store "toy" "train" root)
                  ["toy" "test"] (distilla.store/open-store "toy" "test" root)}
          out (str (fs/path root "run"))
          opts {:train [["toy" "train" train-set]]
                :validation [["toy" "test" (data/load-split "toy" "test" root)]]
                :stores stores :batch-size 2 :accumulation 1 :max-epochs 2 :val-check-interval 1.0
                :output-dir out :peak-lr 1e-2 :warmup-ratio 0.0}]
      (with-open [m (tiny-module)]
        (train/fit! (train/trainer {:memory m :decoder (fake-decoder [0.1]) :tokenizer (h/word-tokenizer)}) opts)
        (is (every? #(fs/exists? (fs/path out % util/safetensors-checkpoint-file-name)) ["epoch-00" "epoch-01"]))
        (is (= {"epoch" 2 "batch" 0} (util/read-json (fs/path out "last" "progress.json"))))
        (let [lines (str/split-lines (slurp (str (fs/path out train/log-file))))]
          (is (= "step,epoch,train_loss,val_loss,lr,grad_norm" (first lines)))
          (is (= 4 (count (rest lines))))))
      (testing "a finished run resumes to nothing more to do, with its weights and step"
        (with-open [m (tiny-module)]
          (let [tr (train/trainer {:memory m :decoder (fake-decoder [0.1]) :tokenizer (h/word-tokenizer)})]
            (train/fit! tr opts)
            (is (= 4 @(get-in tr [:optimizer :step])))
            (with-open [saved (memory/from-pretrained (str (fs/path out "epoch-01")))]
              (is (= (vec (memory/param-floats saved "memory_projection.bias"))
                     (vec (memory/param-floats m "memory_projection.bias")))))))))))

;; --- recipes ---------------------------------------------------------------------------------------------

(deftest recipes-parse
  (doseq [path (fs/glob "configs" "*.yaml")]
    (let [args (train/parse-args ["--config" (str path)])]
      (is (seq (:datasets args)) (str path))
      (is (:output_dir args))
      (is (double? (:learning_rate args)))
      (is (boolean? (:retrieval_init args)))
      (is (zero? (:num_self_attn_layers args))))))

(deftest flags-override-recipe
  (let [args (train/parse-args ["--config" "configs/single_task.yaml" "--datasets" "pmv2" "--learning_rate" "5e-5"])]
    (is (= ["pmv2"] (:datasets args)))
    (is (= 5e-5 (:learning_rate args)))
    (is (= "runs/stage2/epoch-00" (:init_memory args)))))

(deftest unknown-recipe-keys-are-rejected
  (h/with-temp-dir [dir]
    (let [recipe (str (fs/path dir "bad.yaml"))]
      (spit recipe "datasets: [pv4]\noutput_dir: runs/x\nlearnin_rate: 1.0\n")
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"learnin_rate" (train/parse-args ["--config" recipe]))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"datasets" (train/parse-args ["--output_dir" "x"])))))
