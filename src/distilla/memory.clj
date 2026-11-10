(ns distilla.memory
  "The memory module: question-conditioned compression of a variable number
  of document embeddings into a fixed number of memory tokens in the
  decoder's input embedding space. Loads checkpoints trained by the Python
  implementation (memory.pt) or here (memory.safetensors), and trains:
  forward-tape keeps what backward! needs to accumulate parameter gradients
  (distilla.train drives it).

  The forward pass, for one example:

  1. Initial slots: num-memories learned vectors, or with retrieval-init the
     documents most similar to the question (cosine), zero-padded.
  2. Question conditioning: slots cross-attend to the question embedding
     (residual + LayerNorm).
  3. Optional document self-attention: num-self-attn-layers post-norm
     Transformer encoder layers (ReLU feed-forward).
  4. num-cross-attn-layers refinement blocks: pre-norm cross-attention from
     the slots to the normalised documents, then a pre-norm GELU feed-forward
     network, both residual.
  5. Final LayerNorm and a linear projection to the decoder width.

  A module holds its parameters (and gradients) in foreign memory, for BLAS
  and the optimizer kernels, and must be released with `close`."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [distilla.blas :as blas]
            [distilla.safetensors :as st]
            [distilla.tensor :as t]
            [distilla.torch :as torch]
            [distilla.util :as util]
            [jolt.ffi :as ffi]))

(def default-num-memories 16)
(def default-num-heads 8)
(def default-num-self-attn-layers 0)
(def default-num-cross-attn-layers 2)
(def default-dropout 0.1)
(def ffn-expansion 4)

;; --- configuration -----------------------------------------------------------------

(defn config
  "A module configuration with the Python defaults filled in."
  [{:keys [embedding-dim output-dim] :as opts}]
  (when-not (and embedding-dim output-dim)
    (throw (ex-info "a memory config needs :embedding-dim and :output-dim" {:config opts})))
  (merge {:num-memories default-num-memories
          :num-heads default-num-heads
          :num-self-attn-layers default-num-self-attn-layers
          :num-cross-attn-layers default-num-cross-attn-layers
          :dropout default-dropout
          :retrieval-init false}
         (select-keys opts [:embedding-dim :output-dim :num-memories :num-heads :num-self-attn-layers
                            :num-cross-attn-layers :dropout :retrieval-init])))

(def ^:private json-keys
  {:embedding-dim "embedding_dim" :output-dim "output_dim" :num-memories "num_memories"
   :num-heads "num_heads" :num-self-attn-layers "num_self_attn_layers"
   :num-cross-attn-layers "num_cross_attn_layers" :dropout "dropout" :retrieval-init "retrieval_init"})

(def ^:private json-order
  [:embedding-dim :output-dim :num-memories :num-heads :num-self-attn-layers :num-cross-attn-layers
   :dropout :retrieval-init])

(defn config->json [cfg]
  (into (array-map) (for [k json-order] [(json-keys k) (get cfg k)])))

(defn json->config [m]
  (config (into {} (for [[k jk] json-keys :when (contains? m jk)] [k (get m jk)]))))

(defn- count-layers [state pattern]
  (count (into #{} (keep #(second (re-find pattern %)) (keys state)))))

(defn config-from-state-dict
  "Infer the architecture from a state dict. Heads and dropout leave no trace
  in the weights and take their defaults, as does the slot count of a
  retrieval-init module."
  [state]
  (let [retrieval-init (not (contains? state "memory_queries"))]
    (config {:embedding-dim (first (:shape (get state "query_cond_norm.weight")))
             :output-dim (first (:shape (get state "memory_projection.weight")))
             :num-memories (if retrieval-init default-num-memories (second (:shape (get state "memory_queries"))))
             :num-self-attn-layers (count-layers state #"^encoder\.layers\.(\d+)\.")
             :num-cross-attn-layers (count-layers state #"^cross_attn_layers\.(\d+)\.")
             :retrieval-init retrieval-init})))

(defn- attention-keys [prefix]
  (map #(str prefix %) ["in_proj_weight" "in_proj_bias" "out_proj.weight" "out_proj.bias"]))

(defn- norm-keys [prefix] [(str prefix "weight") (str prefix "bias")])

(defn expected-keys
  "The state-dict keys a module with this configuration has."
  [{:keys [retrieval-init num-self-attn-layers num-cross-attn-layers]}]
  (set (concat
        (when-not retrieval-init ["memory_queries"])
        (attention-keys "query_cond_attn.")
        (norm-keys "query_cond_norm.")
        (mapcat (fn [i]
                  (let [p (str "encoder.layers." i ".")]
                    (concat (attention-keys (str p "self_attn."))
                            (norm-keys (str p "linear1.")) (norm-keys (str p "linear2."))
                            (norm-keys (str p "norm1.")) (norm-keys (str p "norm2.")))))
                (range num-self-attn-layers))
        (norm-keys "doc_norm.")
        (mapcat (fn [i]
                  (concat (norm-keys (str "cross_attn_norms." i "."))
                          (attention-keys (str "cross_attn_layers." i "."))
                          (norm-keys (str "ffn_norms." i "."))
                          (norm-keys (str "ffn_layers." i ".0."))
                          (norm-keys (str "ffn_layers." i ".3."))))
                (range num-cross-attn-layers))
        (norm-keys "layer_norm.")
        (norm-keys "memory_projection."))))

;; --- checkpoint loading ----------------------------------------------------------------

(defn- strip-prefix [state prefix]
  (into {} (for [[k v] state] [(subs k (count prefix)) v])))

(defn load-memory-state
  "Memory-module weights from a checkpoint, normalising common wrappers: a
  dict with a state_dict entry (training checkpoints), keys prefixed by the
  owning module (memory.) and by distributed or compiled wrappers.
  Reads memory.pt (torch) or .safetensors."
  [path]
  (let [file (util/checkpoint-file path)
        state (if (str/ends-with? file ".safetensors")
                (st/load-file file)
                (torch/load file))
        state (if (and (map? state) (map? (get state "state_dict"))) (get state "state_dict") state)
        state (reduce (fn [s prefix]
                        (if (and (seq s) (every? #(str/starts-with? % prefix) (keys s)))
                          (strip-prefix s prefix)
                          s))
                      state util/wrapper-prefixes)
        state (if (some #(str/starts-with? % util/memory-prefix) (keys state))
                (strip-prefix (into {} (filter #(str/starts-with? (key %) util/memory-prefix)) state)
                              util/memory-prefix)
                state)]
    (into {} (filter (comp torch/tensor? val)) state)))

;; --- parameters ----------------------------------------------------------------------------
;;
;; Every parameter lives in foreign memory, row-major: matrices feed BLAS
;; directly and the optimizer kernels update them in place. Gradients, when
;; enabled, sit beside them in buffers of the same layout.

(defn param-shapes
  "{name shape} of every parameter of a configuration."
  [{:keys [embedding-dim output-dim num-memories retrieval-init num-self-attn-layers num-cross-attn-layers]}]
  (let [e embedding-dim
        f (* ffn-expansion e)
        attention (fn [p] {(str p "in_proj_weight") [(* 3 e) e] (str p "in_proj_bias") [(* 3 e)]
                           (str p "out_proj.weight") [e e] (str p "out_proj.bias") [e]})
        norm (fn [p] {(str p "weight") [e] (str p "bias") [e]})
        linear (fn [p out in] {(str p "weight") [out in] (str p "bias") [out]})]
    (merge
     (when-not retrieval-init {"memory_queries" [1 num-memories e]})
     (attention "query_cond_attn.")
     (norm "query_cond_norm.")
     (apply merge {} (for [i (range num-self-attn-layers)]
                       (let [p (str "encoder.layers." i ".")]
                         (merge (attention (str p "self_attn."))
                                (linear (str p "linear1.") f e) (linear (str p "linear2.") e f)
                                (norm (str p "norm1.")) (norm (str p "norm2."))))))
     (norm "doc_norm.")
     (apply merge {} (for [i (range num-cross-attn-layers)]
                       (merge (norm (str "cross_attn_norms." i "."))
                              (attention (str "cross_attn_layers." i "."))
                              (norm (str "ffn_norms." i "."))
                              (linear (str "ffn_layers." i ".0.") f e)
                              (linear (str "ffn_layers." i ".3.") e f))))
     (norm "layer_norm.")
     (linear "memory_projection." output-dim e))))

(defrecord MemoryModule [config arena params grads dtype]
  java.io.Closeable
  (close [_] (ffi/close-arena arena)))

(defn num-memories [module] (get-in module [:config :num-memories]))

(defn param-names [module] (sort (keys (:params module))))

(defn param-ptr [module name] (get-in module [:params name :ptr]))

(defn param-floats
  "A parameter's current values, a float array."
  [module name]
  (let [{:keys [ptr n]} (get-in module [:params name])]
    (blas/foreign->floats ptr n)))

(defn set-param!
  "Overwrite a parameter with a float array of its size."
  [module name ^floats data]
  (let [{:keys [ptr n]} (get-in module [:params name])]
    (when-not (= n (alength data))
      (throw (ex-info (str "parameter " name " has " n " values, got " (alength data)) {:name name})))
    (ffi/write-array ptr :float data)))

(defn parameter-count [module] (reduce + 0 (map :n (vals (:params module)))))

(defn- allocate [cfg]
  (let [arena (ffi/shared-arena)
        params (into {} (for [[name shape] (param-shapes cfg)]
                          (let [n (reduce * 1 shape)]
                            [name {:shape shape :n n :ptr (blas/alloc-floats arena n)}])))]
    (->MemoryModule cfg arena params (atom nil) "F32")))

(defn from-state
  "Build a module from a state dict and a configuration. Weights are loaded
  strictly: every expected key must be there and no other."
  [cfg state]
  (let [cfg (config cfg)
        expected (expected-keys cfg)
        actual (set (keys state))]
    (when-not (= expected actual)
      (throw (ex-info "state dict does not match the memory module"
                      {:missing (sort (remove actual expected)) :unexpected (sort (remove expected actual))})))
    (when-not (zero? (mod (:embedding-dim cfg) (:num-heads cfg)))
      (throw (ex-info "embedding-dim must be divisible by num-heads" {:config cfg})))
    (let [m (allocate cfg)]
      (doseq [[name {:keys [data]}] state] (set-param! m name data))
      (assoc m :dtype (:dtype (val (first state)))))))

(defn from-pretrained
  "Load a module from a checkpoint directory or weights file. The
  architecture comes from config.json next to the weights when present and
  is otherwise inferred from the state dict; overrides replace either."
  ([path] (from-pretrained path {}))
  ([path overrides]
   (let [weights (util/checkpoint-file path)
         state (load-memory-state weights)
         config-path (fs/path (fs/parent (fs/absolutize weights)) util/config-file-name)
         cfg (if (fs/exists? config-path)
               (json->config (util/read-json config-path))
               (config-from-state-dict state))]
     (from-state (merge cfg overrides) state))))

(defn state-dict
  "The module's weights as {name {:dtype \"F32\" :shape :data}}."
  [module]
  (into {} (for [[name {:keys [shape]}] (:params module)]
             [name {:dtype "F32" :shape shape :data (param-floats module name)}])))

(defn save
  "Write memory.safetensors (weights, dtype F32 or BF16) and config.json."
  ([module directory] (save module directory "F32"))
  ([module directory dtype]
   (fs/create-dirs directory)
   (st/save-file (update-vals (state-dict module) #(assoc % :dtype dtype))
                 (str (fs/path directory util/safetensors-checkpoint-file-name)))
   (util/write-json (fs/path directory util/config-file-name) (config->json (:config module)))
   (str directory)))

;; --- random numbers and initialisation ---------------------------------------------------

(defn rng
  "A splitmix64 generator state (an atom), for initialisation and dropout."
  [seed]
  (atom (long seed)))

(defn- next-long! [state]
  (let [s (swap! state #(unchecked-add (long %) (unchecked-long -7046029254386353131)))
        z (unchecked-multiply (bit-xor s (unsigned-bit-shift-right s 30)) (unchecked-long -4658895280553007687))
        z (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 27)) (unchecked-long -7723592293110705685))]
    (bit-xor z (unsigned-bit-shift-right z 31))))

(defn uniform!
  "A uniform double in [0, 1)."
  ^double [state]
  (/ (double (unsigned-bit-shift-right (next-long! state) 11)) 9007199254740992.0))

(defn- fill-uniform [state n bound]
  (let [a (float-array n)]
    (dotimes [i n] (aset a i (float (* bound (- (* 2.0 (uniform! state)) 1.0)))))
    a))

(defn init-module
  "A freshly initialised module, with torch's initialisers: Xavier-uniform
  slots, attention input projections and memory projection, zero attention
  biases, Kaiming-uniform (a=sqrt 5) linear layers, unit LayerNorms."
  ([cfg] (init-module cfg 0))
  ([cfg seed]
   (let [cfg (config cfg)
         m (allocate cfg)
         state (rng seed)
         e (:embedding-dim cfg)]
     (doseq [[name {:keys [shape n]}] (sort-by key (:params m))]
       (let [fan-in (if (= 3 (count shape)) (* (nth shape 1) (nth shape 2)) (last shape))
             xavier (fn [fan-in fan-out] (Math/sqrt (/ 6.0 (+ fan-in fan-out))))
             data (cond
                    (= name "memory_queries") (fill-uniform state n (xavier (* (nth shape 1) (nth shape 2)) (nth shape 2)))
                    (or (str/ends-with? name "in_proj_weight") (= name "memory_projection.weight"))
                    (fill-uniform state n (xavier (second shape) (first shape)))
                    (or (str/ends-with? name "in_proj_bias") (str/ends-with? name "out_proj.bias")) (float-array n)
                    (re-find #"norm\d?\.weight$|norms\.\d+\.weight$|^layer_norm\.weight$" name) (float-array n 1.0)
                    (re-find #"norm\d?\.bias$|norms\.\d+\.bias$|^layer_norm\.bias$" name) (float-array n)
                    (str/ends-with? name ".weight") (fill-uniform state n (/ 1.0 (Math/sqrt fan-in)))
                    ;; a Linear bias: bounded by its weight's fan-in
                    :else (let [w (get-in m [:params (str/replace name #"bias$" "weight") :shape])]
                            (fill-uniform state n (/ 1.0 (Math/sqrt (if w (second w) e))))))]
         (set-param! m name data)))
     m)))

(defn load-partial!
  "Copy the tensors of state that the module has (same shape) into it; the
  rest keep their initialisation. Returns {:loaded :missing :unexpected}."
  [module state]
  (let [params (:params module)
        loaded (filterv (fn [[k v]] (and (contains? params k) (= (reduce * 1 (:shape v)) (get-in params [k :n]))))
                        state)]
    (doseq [[k v] loaded] (set-param! module k (:data v)))
    {:loaded (count loaded)
     :missing (vec (sort (remove (set (map key loaded)) (keys params))))
     :unexpected (vec (sort (remove params (keys state))))}))

;; --- gradients ---------------------------------------------------------------------------

(defn ensure-grads!
  "Allocate the gradient buffers (zeroed) the first time; returns {name ptr}."
  [module]
  (or @(:grads module)
      (reset! (:grads module)
              (into {} (for [[name {:keys [n]}] (:params module)]
                         [name (blas/alloc-floats (:arena module) n)])))))

(defn grad-floats [module name]
  (blas/foreign->floats (get @(:grads module) name) (get-in module [:params name :n])))

(defn zero-grads! [module]
  (doseq [[name p] (ensure-grads! module)]
    (ffi/write-array p :float (float-array (get-in module [:params name :n])))))

(defn- add-to-grad!
  "grad[name][offset .. offset+n) += values"
  [module name offset ^floats values]
  (let [p (+ (get (ensure-grads! module) name) (* 4 offset))
        cur (ffi/read-array p :float (alength values))]
    (t/add! cur values)
    (ffi/write-array p :float cur)))

;; --- layers, each a forward that keeps what its backward needs ------------------------------

(defn- weight-view
  "A Linear layer, or row block k of the in_proj of an attention layer."
  ([module prefix]
   (let [{:keys [shape ptr]} (get-in module [:params (str prefix "weight")])]
     {:w ptr :rows (first shape) :cols (second shape) :wname (str prefix "weight") :woff 0
      :bname (str prefix "bias") :boff 0}))
  ([module prefix k]
   (let [e (get-in module [:config :embedding-dim])]
     {:w (+ (param-ptr module (str prefix "in_proj_weight")) (* 4 k e e)) :rows e :cols e
      :wname (str prefix "in_proj_weight") :woff (* k e e)
      :bname (str prefix "in_proj_bias") :boff (* k e)})))

(defn- bias-of [module {:keys [bname boff rows]}]
  (let [p (+ (param-ptr module bname) (* 4 boff))]
    (blas/foreign->floats p rows)))

(defn- linear
  "x (n x cols) through the view: {:y :x :n :view}."
  [module view ^floats x n]
  (let [{:keys [w rows cols]} view
        y (t/add-bias! (blas/matmul-t x n cols w rows) rows (bias-of module view))]
    {:y y :x x :n n :view view}))

(defn- linear-back
  "Accumulate the view's weight and bias gradients; return dx (n x cols)."
  [module {:keys [x n view]} ^floats dy]
  (let [{:keys [w rows cols wname woff bname boff]} view
        grads (ensure-grads! module)]
    (with-open [arena (ffi/confined-arena)]
      (let [dyp (blas/->foreign arena dy)
            xp (blas/->foreign arena x)
            dxp (blas/alloc-floats arena (* n cols))]
        ;; dW += dy^T x ; dx = dy W
        (blas/gemm! true false rows cols n 1.0 dyp rows xp cols 1.0 (+ (get grads wname) (* 4 woff)) cols)
        (blas/gemm! false false n cols rows 1.0 dyp rows w cols 0.0 dxp cols)
        (let [db (float-array rows)]
          (dotimes [i n]
            (dotimes [j rows]
              (aset db j (float (+ (aget db j) (aget dy (+ (* i rows) j)))))))
          (add-to-grad! module bname boff db))
        (blas/foreign->floats dxp (* n cols))))))

(defn- layer-norm-fwd [module prefix ^floats x cols]
  (let [n (quot (alength x) cols)
        gamma (param-floats module (str prefix "weight"))
        beta (param-floats module (str prefix "bias"))
        xhat (float-array (alength x))
        inv (double-array n)
        y (float-array (alength x))]
    (dotimes [i n]
      (let [base (* i cols)
            mean (/ (loop [j 0 s 0.0] (if (< j cols) (recur (inc j) (+ s (aget x (+ base j)))) s)) cols)
            var (/ (loop [j 0 s 0.0] (if (< j cols)
                                       (let [d (- (aget x (+ base j)) mean)] (recur (inc j) (+ s (* d d))))
                                       s))
                   cols)
            iv (/ 1.0 (Math/sqrt (+ var 1e-5)))]
        (aset inv i iv)
        (dotimes [j cols]
          (let [h (* (- (aget x (+ base j)) mean) iv)]
            (aset xhat (+ base j) (float h))
            (aset y (+ base j) (float (+ (* h (aget gamma j)) (aget beta j))))))))
    {:y y :xhat xhat :inv inv :gamma gamma :cols cols :prefix prefix}))

(defn- layer-norm-back [module {:keys [^floats xhat ^doubles inv ^floats gamma cols prefix]} ^floats dy]
  (let [n (quot (alength dy) cols)
        dx (float-array (alength dy))
        dgamma (float-array cols)
        dbeta (float-array cols)]
    (dotimes [i n]
      (let [base (* i cols)
            [s1 s2] (loop [j 0 s1 0.0 s2 0.0]
                      (if (< j cols)
                        (let [g (* (aget dy (+ base j)) (aget gamma j))]
                          (recur (inc j) (+ s1 g) (+ s2 (* g (aget xhat (+ base j))))))
                        [s1 s2]))
            iv (aget inv i)]
        (dotimes [j cols]
          (let [k (+ base j)
                g (* (aget dy k) (aget gamma j))]
            (aset dgamma j (float (+ (aget dgamma j) (* (aget dy k) (aget xhat k)))))
            (aset dbeta j (float (+ (aget dbeta j) (aget dy k))))
            (aset dx k (float (* (/ iv cols) (- (* cols g) s1 (* (aget xhat k) s2)))))))))
    (add-to-grad! module (str prefix "weight") 0 dgamma)
    (add-to-grad! module (str prefix "bias") 0 dbeta)
    dx))

(defn- dropout-fwd
  "Inverted dropout in training mode: {:y :mask} (mask nil when inactive)."
  [{:keys [train? p rng]} ^floats x]
  (if (and train? (pos? p))
    (let [keep (/ 1.0 (- 1.0 p))
          mask (float-array (alength x))
          y (float-array (alength x))]
      (dotimes [i (alength x)]
        (when (>= (uniform! rng) p)
          (aset mask i (float keep))
          (aset y i (float (* keep (aget x i))))))
      {:y y :mask mask})
    {:y x :mask nil}))

(defn- dropout-back [{:keys [^floats mask]} ^floats dy]
  (if mask
    (let [dx (float-array (alength dy))]
      (dotimes [i (alength dy)] (aset dx i (float (* (aget mask i) (aget dy i)))))
      dx)
    dy))

(def ^:private inv-sqrt2 (/ 1.0 (Math/sqrt 2.0)))
(def ^:private inv-sqrt-2pi (/ 1.0 (Math/sqrt (* 2.0 Math/PI))))

(defn- gelu-back [^floats x ^floats dy]
  (let [dx (float-array (alength x))]
    (dotimes [i (alength x)]
      (let [v (double (aget x i))
            cdf (* 0.5 (+ 1.0 (t/erf (* v inv-sqrt2))))
            pdf (* inv-sqrt-2pi (Math/exp (* -0.5 v v)))]
        (aset dx i (float (* (aget dy i) (+ cdf (* v pdf)))))))
    dx))

(defn- relu-back [^floats x ^floats dy]
  (let [dx (float-array (alength x))]
    (dotimes [i (alength x)] (when (pos? (aget x i)) (aset dx i (aget dy i))))
    dx))

(defn- attention
  "nn.MultiheadAttention (batch first): queries (m x E) attend to keys and
  values (n x E); in training mode the attention weights are dropped out."
  [module ctx prefix ^floats query m ^floats kv n]
  (let [e (get-in module [:config :embedding-dim])
        heads (get-in module [:config :num-heads])
        hd (quot e heads)
        scale (/ 1.0 (Math/sqrt hd))
        lq (linear module (weight-view module prefix 0) query m)
        lk (linear module (weight-view module prefix 1) kv n)
        lv (linear module (weight-view module prefix 2) kv n)
        probs (transient [])
        drops (transient [])
        o (with-open [arena (ffi/confined-arena)]
            (let [qp (blas/->foreign arena (:y lq))
                  kp (blas/->foreign arena (:y lk))
                  vp (blas/->foreign arena (:y lv))
                  sp (blas/alloc-floats arena (* m n))
                  op (blas/alloc-floats arena (* m e))]
              (dotimes [h heads]
                (let [off (* 4 h hd)]
                  (blas/gemm! false true m n hd scale (+ qp off) e (+ kp off) e 0.0 sp n)
                  (let [s (blas/foreign->floats sp (* m n))
                        _ (t/softmax-rows! s n nil)
                        d (dropout-fwd ctx s)]
                    (conj! probs s)
                    (conj! drops d)
                    (ffi/write-array sp :float (:y d)))
                  (blas/gemm! false false m hd n 1.0 sp n (+ vp off) e 0.0 (+ op off) e)))
              (blas/foreign->floats op (* m e))))
        lo (linear module (weight-view module (str prefix "out_proj.")) o m)]
    {:y (:y lo) :m m :n n :lq lq :lk lk :lv lv :lo lo :probs (persistent! probs) :drops (persistent! drops)
     :scale scale :prefix prefix}))

(defn- attention-back
  "Backward of attention: [d-query d-kv]."
  [module {:keys [m n lq lk lv lo probs drops scale]} ^floats dy]
  (let [e (get-in module [:config :embedding-dim])
        heads (get-in module [:config :num-heads])
        hd (quot e heads)
        d-o (linear-back module lo dy)
        [dq dk dv]
        (with-open [arena (ffi/confined-arena)]
          (let [qp (blas/->foreign arena (:y lq))
                kp (blas/->foreign arena (:y lk))
                vp (blas/->foreign arena (:y lv))
                dop (blas/->foreign arena d-o)
                pp (blas/alloc-floats arena (* m n))
                tp (blas/alloc-floats arena (* m n))
                dqp (blas/alloc-floats arena (* m e))
                dkp (blas/alloc-floats arena (* n e))
                dvp (blas/alloc-floats arena (* n e))]
            (dotimes [h heads]
              (let [off (* 4 h hd)
                    ^floats p (nth probs h)
                    drop (nth drops h)]
                ;; dV_h = Pd^T dO_h ; dPd = dO_h V_h^T
                (ffi/write-array pp :float (:y drop))
                (blas/gemm! true false n hd m 1.0 pp n (+ dop off) e 0.0 (+ dvp off) e)
                (blas/gemm! false true m n hd 1.0 (+ dop off) e (+ vp off) e 0.0 tp n)
                (let [dp (dropout-back drop (blas/foreign->floats tp (* m n)))
                      ds (float-array (* m n))]
                  ;; softmax backward, then the scale
                  (dotimes [i m]
                    (let [base (* i n)
                          dot (loop [j 0 s 0.0] (if (< j n)
                                                  (recur (inc j) (+ s (* (aget dp (+ base j)) (aget p (+ base j)))))
                                                  s))]
                      (dotimes [j n]
                        (aset ds (+ base j) (float (* scale (aget p (+ base j)) (- (aget dp (+ base j)) dot)))))))
                  (ffi/write-array tp :float ds))
                ;; dQ_h = dS K_h ; dK_h = dS^T Q_h
                (blas/gemm! false false m hd n 1.0 tp n (+ kp off) e 0.0 (+ dqp off) e)
                (blas/gemm! true false n hd m 1.0 tp n (+ qp off) e 0.0 (+ dkp off) e)))
            [(blas/foreign->floats dqp (* m e)) (blas/foreign->floats dkp (* n e)) (blas/foreign->floats dvp (* n e))]))
        d-query (linear-back module lq dq)
        d-kv (t/add! (linear-back module lk dk) (linear-back module lv dv))]
    [d-query d-kv]))

(defn- encoder-layer
  "One post-norm nn.TransformerEncoderLayer (ReLU)."
  [module ctx i ^floats x n]
  (let [e (get-in module [:config :embedding-dim])
        p (str "encoder.layers." i ".")
        sa (attention module ctx (str p "self_attn.") x n x n)
        d1 (dropout-fwd ctx (:y sa))
        n1 (layer-norm-fwd module (str p "norm1.") (t/add! (t/copy-floats x) (:y d1)) e)
        l1 (linear module (weight-view module (str p "linear1.")) (:y n1) n)
        h (t/relu! (t/copy-floats (:y l1)))
        dh (dropout-fwd ctx h)
        l2 (linear module (weight-view module (str p "linear2.")) (:y dh) n)
        d2 (dropout-fwd ctx (:y l2))
        n2 (layer-norm-fwd module (str p "norm2.") (t/add! (t/copy-floats (:y n1)) (:y d2)) e)]
    {:y (:y n2) :sa sa :d1 d1 :n1 n1 :l1 l1 :dh dh :l2 l2 :d2 d2 :n2 n2}))

(defn- encoder-layer-back [module {:keys [sa d1 n1 l1 dh l2 d2 n2]} ^floats dy]
  (let [d-in2 (layer-norm-back module n2 dy)               ; d(n1 + d2)
        d-l2 (dropout-back d2 d-in2)
        d-h (dropout-back dh (linear-back module l2 d-l2))
        d-l1 (relu-back (:y l1) d-h)
        d-n1 (t/add! (t/copy-floats d-in2) (linear-back module l1 d-l1))
        d-in1 (layer-norm-back module n1 d-n1)             ; d(x + d1)
        [dq dkv] (attention-back module sa (dropout-back d1 d-in1))]
    (t/add! (t/add! (t/copy-floats d-in1) dq) dkv)))

;; --- the module -----------------------------------------------------------------------------

(defn initial-slots
  "The slots before question conditioning: learned vectors, or with
  retrieval-init the documents most similar to the question, zero-padded.
  docs is an n x E matrix; mask, when given, marks its padding rows (true).
  As in the Python implementation, the top-k runs over all n rows with the
  padding scored -inf, so with fewer real documents than slots a padding row
  can fill a slot (the last padding row first, as CPU torch.topk breaks
  the tie)."
  ([module docs question] (initial-slots module docs nil question))
  ([module docs mask ^floats question]
   (let [{:keys [num-memories embedding-dim retrieval-init]} (:config module)]
     (if-not retrieval-init
       (param-floats module "memory_queries")
       (let [n (:rows docs)
             mask (when mask (vec mask))
             normed (t/l2-normalize-rows (:data docs) embedding-dim)
             qn (t/l2-normalize-rows question embedding-dim)
             scores (vec (for [i (range n)]
                           (if (and mask (nth mask i))
                             Double/NEGATIVE_INFINITY
                             (loop [j 0 s 0.0]
                               (if (< j embedding-dim)
                                 (recur (inc j) (+ s (* (aget ^floats normed (+ (* i embedding-dim) j)) (aget qn j))))
                                 s)))))
             k (min num-memories n)
             top (take k (sort-by (fn [i] (let [s (nth scores i)]
                                            (if (Double/isInfinite s) [1 0.0 (- i)] [0 (- s) i])))
                                  (range n)))
             out (float-array (* num-memories embedding-dim))]
         (doseq [[slot i] (map vector (range) top)]
           (System/arraycopy (:data docs) (* i embedding-dim) out (* slot embedding-dim) embedding-dim))
         out)))))

(defn forward-tape
  "The forward pass of one example, keeping what `backward!` needs.
  docs: the example's n x E documents (no padding); question: its E floats.
  opts: :train? (dropout on, default false), :rng (a `rng`, needed with
  dropout), :slot-docs / :slot-mask (the padded documents retrieval-init
  slots are drawn from). Returns {:output matrix :tape ...}."
  [module docs ^floats question {:keys [train? rng slot-docs slot-mask]}]
  (let [{:keys [embedding-dim output-dim num-memories num-self-attn-layers num-cross-attn-layers dropout]}
        (:config module)
        e embedding-dim
        m num-memories
        n (:rows docs)
        ctx {:train? (boolean train?) :p (if train? dropout 0.0) :rng rng}
        slots0 (initial-slots module (or slot-docs docs) (when slot-docs slot-mask) question)
        qc (attention module ctx "query_cond_attn." slots0 m question 1)
        qn (layer-norm-fwd module "query_cond_norm." (t/add! (t/copy-floats slots0) (:y qc)) e)
        encoders (reduce (fn [acc i] (conj acc (encoder-layer module ctx i (if (seq acc) (:y (peek acc)) (:data docs)) n)))
                         [] (range num-self-attn-layers))
        doc-x (if (seq encoders) (:y (peek encoders)) (:data docs))
        dn (layer-norm-fwd module "doc_norm." doc-x e)
        [slots blocks]
        (reduce
         (fn [[slots blocks] i]
           (let [cn (layer-norm-fwd module (str "cross_attn_norms." i ".") slots e)
                 ca (attention module ctx (str "cross_attn_layers." i ".") (:y cn) m (:y dn) n)
                 ad (dropout-fwd ctx (:y ca))
                 slots (t/add! (t/copy-floats slots) (:y ad))
                 fn' (layer-norm-fwd module (str "ffn_norms." i ".") slots e)
                 l0 (linear module (weight-view module (str "ffn_layers." i ".0.")) (:y fn') m)
                 g (t/gelu! (t/copy-floats (:y l0)))
                 gd (dropout-fwd ctx g)
                 l3 (linear module (weight-view module (str "ffn_layers." i ".3.")) (:y gd) m)
                 fd (dropout-fwd ctx (:y l3))]
             [(t/add! (t/copy-floats slots) (:y fd))
              (conj blocks {:cn cn :ca ca :ad ad :fn fn' :l0 l0 :gd gd :l3 l3 :fd fd})]))
         [(:y qn) []] (range num-cross-attn-layers))
        ln (layer-norm-fwd module "layer_norm." slots e)
        proj (linear module (weight-view module "memory_projection.") (:y ln) m)]
    {:output (t/matrix m output-dim (:y proj))
     :tape {:qc qc :qn qn :encoders encoders :dn dn :blocks blocks :ln ln :proj proj :n n}}))

(defn backward!
  "Accumulate the parameter gradients of one example given d(loss)/d(output)
  (a num-memories x output-dim matrix) and the tape of its forward-tape."
  [module {:keys [qc qn encoders dn blocks ln proj n]} d-output]
  (ensure-grads! module)
  (let [e (get-in module [:config :embedding-dim])
        d-slots (layer-norm-back module ln (linear-back module proj (:data d-output)))
        [d-slots d-dn]
        (reduce
         (fn [[d-slots d-dn] {:keys [cn ca ad fn l0 gd l3 fd]}]
           (let [;; slots_out = s1 + fd(l3(gd(gelu(l0(ffn_norm(s1))))))
                 d-l3 (dropout-back fd d-slots)
                 d-g (dropout-back gd (linear-back module l3 d-l3))
                 d-l0 (gelu-back (:y l0) d-g)
                 d-s1 (t/add! (t/copy-floats d-slots) (layer-norm-back module fn (linear-back module l0 d-l0)))
                 ;; s1 = s0 + ad(attention(cross_norm(s0), docs))
                 [dq dkv] (attention-back module ca (dropout-back ad d-s1))
                 d-s0 (t/add! (t/copy-floats d-s1) (layer-norm-back module cn dq))]
             [d-s0 (if d-dn (t/add! d-dn dkv) dkv)]))
         [d-slots nil] (reverse blocks))
        d-sum (layer-norm-back module qn d-slots)          ; d(slots0 + qc)
        [dq0 _] (attention-back module qc d-sum)]
    (when (contains? (:params module) "memory_queries")
      (add-to-grad! module "memory_queries" 0 (t/add! (t/copy-floats d-sum) dq0)))
    ;; the documents are inputs; their path matters only for the parameters on it
    (when d-dn
      (let [d-docs (layer-norm-back module dn d-dn)]
        (reduce (fn [d enc] (encoder-layer-back module enc d)) d-docs (reverse encoders))))
    nil))

(defn forward
  "Compress one example's documents into memory tokens (inference).
  docs: an n x E matrix of the example's documents (no padding);
  question: its E-dimensional embedding (float array).
  opts :slot-docs / :slot-mask give the padded documents the retrieval-init
  slots are drawn from (see initial-slots); by default the documents.
  Returns a num-memories x output-dim matrix."
  ([module docs question] (forward module docs question nil))
  ([module docs question opts]
   (:output (forward-tape module docs question (assoc opts :train? false)))))

(defn valid-docs
  "The rows of a padded document matrix that are not padding."
  [docs row-mask]
  (if row-mask
    (t/select-rows docs (keep-indexed (fn [i pad?] (when-not pad? i)) row-mask))
    docs))

(defn forward-batch
  "The batched form: doc-embeds is a sequence of n_max x E matrices, mask a
  sequence of boolean sequences (true = padding) or nil, questions a
  sequence of float arrays. Padding rows are dropped before the forward pass,
  which is what the key padding mask does in the Python implementation.
  Returns one num-memories x output-dim matrix per example."
  [module doc-embeds mask questions]
  (mapv (fn [docs row-mask question]
          (forward module (valid-docs docs row-mask) question {:slot-docs docs :slot-mask row-mask}))
        doc-embeds (or mask (repeat nil)) questions))
