(ns distilla.llama
  "The decoder and the encoder on llama.cpp, through the native/distilla_llama
  shim (built by `jolt native`).

  - `load-model` loads a GGUF file; models are Closeable.
  - `tokenizer` is the model's vocabulary behind distilla.tokenizer/Tokenizer.
  - `decoder` generates from a prompt given as segments of token ids and of
    input embeddings, which is how memory tokens enter the prompt.
  - `encoder` embeds texts (pooled, L2-normalised) behind
    distilla.encoder/Encoder, for embedding documents and questions and for
    choice scoring."
  (:require [clojure.string :as str]
            [distilla.encoder :as encoder]
            [distilla.tensor :as t]
            [distilla.tokenizer :as tok]
            [jolt.ffi :as ffi]))

(ffi/defcfn dl-quiet "dl_quiet" [:int] :void)
(ffi/defcfn dl-model-load "dl_model_load" [:pointer :int32 :int32] :pointer :blocking)
(ffi/defcfn dl-model-free "dl_model_free" [:pointer] :void)
(ffi/defcfn dl-model-n-embd "dl_model_n_embd" [:pointer] :int32)
(ffi/defcfn dl-model-n-ctx-train "dl_model_n_ctx_train" [:pointer] :int32)
(ffi/defcfn dl-token-eos "dl_token_eos" [:pointer] :int32)
(ffi/defcfn dl-token-is-eog "dl_token_is_eog" [:pointer :int32] :int32)
(ffi/defcfn dl-tokenize "dl_tokenize" [:pointer :pointer :int32 :pointer :int32 :int32 :int32] :int32)
(ffi/defcfn dl-detokenize "dl_detokenize" [:pointer :pointer :int32 :pointer :int32 :int32] :int32)
(ffi/defcfn dl-ctx-new "dl_ctx_new" [:pointer :int32 :int32 :int32 :int32 :int32 :int32 :float :int32] :pointer :blocking)
(ffi/defcfn dl-ctx-free "dl_ctx_free" [:pointer] :void)
(ffi/defcfn dl-ctx-n-ctx "dl_ctx_n_ctx" [:pointer] :int32)
(ffi/defcfn dl-ctx-clear "dl_ctx_clear" [:pointer] :void)
(ffi/defcfn dl-kv-truncate "dl_kv_truncate" [:pointer :int32] :int32)
(ffi/defcfn dl-decode-tokens "dl_decode_tokens" [:pointer :pointer :int32 :int32 :int32] :int32 :blocking)
(ffi/defcfn dl-decode-embd "dl_decode_embd" [:pointer :pointer :int32 :int32 :int32] :int32 :blocking)
(ffi/defcfn dl-generate "dl_generate" [:pointer :int32 :int32 :float :float :uint :pointer :int32 :pointer :pointer] :int32 :blocking)
(ffi/defcfn dl-embed "dl_embed" [:pointer :pointer :pointer :pointer :int32 :pointer] :int32 :blocking)

(def pooling-types {:default -1 :none 0 :mean 1 :cls 2 :last 3})

(defn available? []
  (some? (ffi/find-symbol "dl_model_load")))

(defn- check-available []
  (when-not (available?)
    (throw (ex-info "the llama.cpp shim is not loaded: run `jolt native` to build native/libdistilla_llama" {}))))

(defn quiet!
  "Silence llama.cpp's logging (errors still print), or let it through."
  [quiet?]
  (check-available)
  (dl-quiet (if quiet? 1 0)))

;; --- model ---------------------------------------------------------------------------

(defrecord Model [ptr path n-embd n-ctx-train eos]
  java.io.Closeable
  (close [_] (dl-model-free ptr)))

(defn load-model
  "Load a GGUF model. opts: :n-gpu-layers (default -1, every layer),
  :vocab-only (just the tokenizer)."
  ([path] (load-model path {}))
  ([path {:keys [n-gpu-layers vocab-only] :or {n-gpu-layers -1}}]
   (check-available)
   (let [ptr (ffi/with-c-string [p (str path)] (dl-model-load p n-gpu-layers (if vocab-only 1 0)))]
     (when (ffi/null? ptr)
       (throw (ex-info (str "llama.cpp could not load " path) {:path (str path)})))
     (->Model ptr (str path) (dl-model-n-embd ptr) (dl-model-n-ctx-train ptr) (dl-token-eos ptr)))))

(defn tokenize
  "Token ids of text. add-special adds BOS/EOS as the model's metadata asks;
  parse-special lets special tokens like <|im_start|> through as themselves."
  ([model text] (tokenize model text false true))
  ([model text add-special? parse-special?]
   (let [bytes (.getBytes ^String text "UTF-8")
         n (alength bytes)]
     (with-open [arena (ffi/confined-arena)]
       (let [tp (ffi/alloc arena (inc n))
             _ (when (pos? n) (ffi/write-array tp bytes))
             run (fn [cap]
                   (let [out (ffi/alloc arena (* 4 (max 1 cap)))
                         got (dl-tokenize (:ptr model) tp n out cap (if add-special? 1 0) (if parse-special? 1 0))]
                     [got out]))
             [got out] (run (+ n 16))
             [got out] (if (neg? got) (run (- got)) [got out])]
         (when (neg? got) (throw (ex-info "tokenization failed" {:text text})))
         (if (zero? got) [] (vec (ffi/read-array out :int32 got))))))))

(defn detokenize
  "Text of token ids, special tokens dropped unless render-special?."
  ([model ids] (detokenize model ids false))
  ([model ids render-special?]
   (if (empty? ids)
     ""
     (with-open [arena (ffi/confined-arena)]
       (let [ip (ffi/alloc arena (* 4 (count ids)))
             _ (ffi/write-array ip :int32 (int-array ids))
             run (fn [cap]
                   (let [out (ffi/alloc arena (max 1 cap))]
                     [(dl-detokenize (:ptr model) ip (count ids) out cap (if render-special? 1 0)) out]))
             [got out] (run (+ 64 (* 8 (count ids))))
             [got out] (if (neg? got) (run (- got)) [got out])]
         (if (pos? got) (String. ^bytes (ffi/read-array out got) "UTF-8") ""))))))

;; --- tokenizer ---------------------------------------------------------------------------

(defrecord LlamaTokenizer [model template]
  tok/Tokenizer
  (encode-text [_ text] (tokenize model text false true))
  (decode-ids [_ ids] (detokenize model (remove neg? ids) false))
  (pad-id [_] (:eos model))
  (memory-token-id [_] -1)
  (chat-template [_] template))

(defn tokenizer
  "The model's tokenizer with a chat template (default Qwen3's, thinking off)."
  ([model] (tokenizer model tok/qwen3))
  ([model template] (->LlamaTokenizer model template)))

;; --- decoder -----------------------------------------------------------------------------

(defrecord Decoder [model ctx n-ctx kv]
  java.io.Closeable
  (close [_] (dl-ctx-free ctx)))

(defn- new-ctx [model {:keys [n-ctx n-batch n-threads embeddings? pooling n-seq-max yarn-factor yarn-orig-ctx]
                       :or {n-batch 512 n-threads 0 pooling :default n-seq-max 1 yarn-factor 0.0}}]
  (let [ctx (dl-ctx-new (:ptr model) n-ctx n-batch n-threads (if embeddings? 1 0) (pooling-types pooling)
                        n-seq-max (float yarn-factor) (or yarn-orig-ctx (:n-ctx-train model)))]
    (when (ffi/null? ctx)
      (throw (ex-info "llama.cpp could not create a context" {:n-ctx n-ctx :n-batch n-batch})))
    ctx))

(defn decoder
  "A generation context on model. opts: :n-ctx (context length, default
  8192), :n-batch (512), :n-threads (0 = llama.cpp's default), :yarn-factor
  and :yarn-orig-ctx for YaRN RoPE scaling."
  ([model] (decoder model {}))
  ([model opts]
   (let [opts (merge {:n-ctx 8192} opts)
         ctx (new-ctx model opts)]
     (->Decoder model ctx (dl-ctx-n-ctx ctx) (atom [])))))

;; The KV cache as a continuation store: a decoder remembers what its cache
;; holds, one key per position (a token id, or a hash of an embedding row),
;; and a new prompt decodes only what follows the longest common prefix. The
;; constant system-prompt prefix of every Memorilla prompt is decoded once;
;; a multi-turn agent, whose every prompt extends the previous one with the
;; reply and the next observation, re-decodes only the new turn. A reused
;; cache gives the same logits as decoding from scratch (the append law),
;; which the tests check.

(defn- row-hash
  "A 64-bit hash of an embedding row's bits."
  [^floats data offset cols]
  (loop [j 0 h (unchecked-long -3750763034362895579)]
    (if (< j cols)
      (recur (inc j) (unchecked-multiply (bit-xor h (Float/floatToRawIntBits (aget data (+ offset j))))
                                         (unchecked-long 1099511628211)))
      [:embedding h])))

(defn- segment-keys [segments]
  (vec (mapcat (fn [[kind x]]
                 (if (= kind :tokens)
                   x
                   (for [i (range (:rows x))] (row-hash (:data x) (* i (:cols x)) (:cols x)))))
               segments)))

(defn- drop-positions
  "The segments without their first n positions."
  [segments n]
  (loop [segs segments n n acc []]
    (if-let [[kind x :as seg] (first segs)]
      (let [len (if (= kind :tokens) (count x) (:rows x))]
        (cond
          (zero? n) (recur (rest segs) 0 (conj acc seg))
          (>= n len) (recur (rest segs) (- n len) acc)
          (= kind :tokens) (recur (rest segs) 0 (conj acc [:tokens (subvec x n)]))
          :else (recur (rest segs) 0 (conj acc [:embeddings (t/select-rows x (range n len))]))))
      acc)))

(defn- decode-segments!
  "Decode segments into the cache from position pos0, logits kept for the last."
  [{:keys [model ctx]} segments pos0]
  (with-open [arena (ffi/confined-arena)]
    (reduce
     (fn [pos [i [kind x]]]
       (let [last? (= i (dec (count segments)))
             rc (if (= kind :tokens)
                  (let [p (ffi/alloc arena (* 4 (count x)))]
                    (ffi/write-array p :int32 (int-array x))
                    (dl-decode-tokens ctx p (count x) pos (if last? 1 0)))
                  (do (when-not (= (:cols x) (:n-embd model))
                        (throw (ex-info "embedding width does not match the decoder"
                                        {:cols (:cols x) :n-embd (:n-embd model)})))
                      (let [p (ffi/alloc arena (* 4 (alength ^floats (:data x))))]
                        (ffi/write-array p :float (:data x))
                        (dl-decode-embd ctx p (:rows x) pos (if last? 1 0)))))]
         (when-not (zero? rc)
           (throw (ex-info (str "llama_decode failed (" rc ")") {:rc rc :segment kind})))
         (+ pos (if (= kind :tokens) (count x) (:rows x)))))
     pos0 (map-indexed vector segments))))

(defn- prepare-prompt!
  "Bring the cache to the prompt, reusing its common prefix (unless reuse? is
  false); returns the prompt's keys. At least the last position is decoded,
  since its logits are not kept in the cache."
  [{:keys [ctx kv] :as dc} segments reuse?]
  (let [ks (segment-keys segments)
        n (count ks)
        cached @kv
        common (if reuse?
                 (loop [i 0] (if (and (< i n) (< i (count cached)) (= (nth ks i) (nth cached i))) (recur (inc i)) i))
                 0)
        keep (min common (dec n))
        keep (if (or (= keep (count cached)) (= 1 (dl-kv-truncate ctx keep))) keep (do (dl-ctx-clear ctx) 0))]
    (reset! kv (subvec ks 0 keep))
    (decode-segments! dc (drop-positions segments keep) keep)
    (reset! kv ks)
    ks))

(defn generate
  "Generate from a prompt made of segments, each [:tokens ids] or
  [:embeddings matrix] (rows of the decoder's width), decoded in order into
  one sequence. opts: :max-new-tokens (256) :temperature (0 = greedy) :top-p
  (1.0) :seed, :reuse-prefix (true: decode only what follows the prefix the
  cache already holds). Returns the generated text without special tokens;
  with :tokens? true, {:text :tokens :stop-token :stop-reason}: :stop-token is
  the end-of-generation token that ended it (not in :tokens) or nil, and
  :stop-reason \"stop\" or \"length\"."
  [{:keys [model ctx n-ctx kv] :as dc} segments
   {:keys [max-new-tokens temperature top-p seed reuse-prefix tokens?]
    :or {max-new-tokens 256 temperature 0.0 top-p 1.0 seed 0 reuse-prefix true}}]
  (let [segments (vec (remove (fn [[kind x]] (if (= kind :tokens) (empty? x) (zero? (:rows x)))) segments))
        total (reduce + 0 (map (fn [[kind x]] (if (= kind :tokens) (count x) (:rows x))) segments))]
    (when (empty? segments) (throw (ex-info "empty prompt" {})))
    (when (>= total n-ctx)
      (throw (ex-info (str "prompt of " total " tokens does not fit the context of " n-ctx) {:tokens total :n-ctx n-ctx})))
    (let [ks (prepare-prompt! dc segments reuse-prefix)]
      (with-open [arena (ffi/confined-arena)]
        (let [cap (max 1 max-new-tokens)
              out (ffi/alloc arena (* 4 cap))
              decoded (ffi/alloc arena 4)
              stop (ffi/alloc arena 4)
              n (dl-generate ctx total max-new-tokens (float temperature) (float top-p) seed out cap decoded stop)]
          (when (neg? n) (reset! kv []) (throw (ex-info "generation failed" {:rc n})))
          (let [ids (if (pos? n) (vec (ffi/read-array out :int32 n)) [])]
            (reset! kv (into ks (take (ffi/read decoded :int32) ids)))
            (let [text (detokenize model ids false)
                  stop-token (let [st (ffi/read stop :int32)] (when (>= st 0) st))]
              (if tokens?
                {:text text :tokens ids :stop-token stop-token :stop-reason (if stop-token "stop" "length")}
                text))))))))

;; --- encoder -------------------------------------------------------------------------------

(defrecord LlamaEncoder [model ctx n-ctx n-batch n-seq-max append-eos?]
  java.io.Closeable
  (close [_] (dl-ctx-free ctx))
  encoder/Encoder
  (embed [_ texts]
    (let [dim (:n-embd model)
          limit (min n-ctx n-batch)
          seqs (mapv (fn [text]
                       (let [ids (tokenize model text true true)
                             ids (if (and append-eos? (not= (peek ids) (:eos model))) (conj ids (:eos model)) ids)]
                         ;; keep the end (with its EOS) when a text is too long: the last token is pooled
                         (if (> (count ids) limit) (subvec ids (- (count ids) limit)) ids)))
                     texts)
          batches (loop [i 0 cur [] tokens 0 acc []]
                    (if (= i (count seqs))
                      (if (seq cur) (conj acc cur) acc)
                      (let [n (count (nth seqs i))]
                        (if (and (seq cur) (or (> (+ tokens n) n-batch) (= (count cur) n-seq-max)))
                          (recur i [] 0 (conj acc cur))
                          (recur (inc i) (conj cur i) (+ tokens n) acc)))))
          out (float-array (* (count texts) dim))]
      (doseq [batch batches]
        (with-open [arena (ffi/confined-arena)]
          (let [ids (vec (mapcat seqs batch))
                lengths (mapv #(count (nth seqs %)) batch)
                offsets (vec (butlast (reductions + 0 lengths)))
                tp (ffi/alloc arena (* 4 (max 1 (count ids))))
                op (ffi/alloc arena (* 4 (count batch)))
                lp (ffi/alloc arena (* 4 (count batch)))
                vp (ffi/alloc arena (* 4 dim (count batch)))]
            (ffi/write-array tp :int32 (int-array ids))
            (ffi/write-array op :int32 (int-array offsets))
            (ffi/write-array lp :int32 (int-array lengths))
            (let [rc (dl-embed ctx tp op lp (count batch) vp)]
              (when-not (zero? rc) (throw (ex-info (str "embedding failed (" rc ")") {:rc rc}))))
            (let [vecs (t/l2-normalize-rows (ffi/read-array vp :float (* dim (count batch))) dim)]
              (doseq [[k i] (map-indexed vector batch)]
                (System/arraycopy vecs (* k dim) out (* i dim) dim))))))
      (t/matrix (count texts) dim out))))

(defn encoder
  "An embedding context on model: last-token pooling by default (Qwen3-Embedding).
  opts: :n-ctx (8192) :n-batch (8192, tokens per embedding batch) :n-seq-max
  (64, texts per batch) :pooling (:default :mean :cls :last) :append-eos
  (true: end every text with EOS, as the Qwen3-Embedding tokenizer does)."
  ([model] (encoder model {}))
  ([model {:keys [n-ctx n-batch n-seq-max pooling append-eos n-threads]
           :or {n-ctx 8192 n-batch 8192 n-seq-max 64 pooling :last append-eos true n-threads 0}}]
   (let [ctx (new-ctx model {:n-ctx n-ctx :n-batch n-batch :n-threads n-threads :embeddings? true
                             :pooling pooling :n-seq-max n-seq-max})]
     (->LlamaEncoder model ctx (dl-ctx-n-ctx ctx) n-batch n-seq-max append-eos))))

;; --- logits (for parity checks) -----------------------------------------------------------

(ffi/defcfn dl-logits-last "dl_logits_last" [:pointer :pointer] :int32)
(ffi/defcfn dl-model-n-vocab "dl_model_n_vocab" [:pointer] :int32)

(defn n-vocab [model] (dl-model-n-vocab (:ptr model)))

(defn last-logits
  "Decode segments (as generate does, reusing the cached prefix unless
  :reuse-prefix false) and return the last position's logits."
  ([dc segments] (last-logits dc segments {}))
  ([{:keys [model ctx] :as dc} segments {:keys [reuse-prefix] :or {reuse-prefix true}}]
   (prepare-prompt! dc (vec segments) reuse-prefix)
   (with-open [arena (ffi/confined-arena)]
     (let [n (n-vocab model)
           out (ffi/alloc arena (* 4 n))]
       (when (neg? (dl-logits-last ctx out)) (throw (ex-info "no logits" {})))
       (ffi/read-array out :float n)))))

;; --- the training graph (native/distilla_train.c) -----------------------------------------

(ffi/defcfn dt-last-error "dt_last_error" [] :pointer)
(ffi/defcfn dt-model-load "dt_model_load" [:pointer :int32 :int32 :int32] :pointer :blocking)
(ffi/defcfn dt-model-free "dt_model_free" [:pointer] :void)
(ffi/defcfn dt-model-n-embd "dt_model_n_embd" [:pointer] :int32)
(ffi/defcfn dt-model-n-vocab "dt_model_n_vocab" [:pointer] :int32)
(ffi/defcfn dt-model-on-gpu "dt_model_on_gpu" [:pointer] :int32)
(ffi/defcfn dt-step "dt_step" [:pointer :pointer :pointer :pointer :int32 :pointer :int32 :pointer :pointer :int32]
  :int32 :blocking)

(ffi/defcfn dt-set-chunking "dt_set_chunking" [:pointer :int32 :int32] :void)

(defn- dt-error [] (ffi/ptr->string (dt-last-error)))

(defn set-chunking!
  "How a grad-model trades memory for recomputation: sequences longer than
  :full-max positions (default 1024) take the layer-wise backward, which
  rebuilds one layer at a time for :chunk query rows (default 256)."
  [{:keys [ptr]} {:keys [chunk full-max]}]
  (dt-set-chunking ptr (or chunk 0) (if (some? full-max) full-max -1)))

(defrecord GradModel [ptr path n-embd n-vocab on-gpu?]
  java.io.Closeable
  (close [_] (dt-model-free ptr)))

(defn grad-model
  "Load a decoder GGUF for training: its loss and the gradient with respect to
  the memory tokens. opts: :gpu? (true) :n-threads (0 = default) :f32-weights
  (false; true keeps the backward pass on a CUDA GPU at 4 bytes a weight)."
  ([path] (grad-model path {}))
  ([path {:keys [gpu? n-threads f32-weights] :or {gpu? true n-threads 0 f32-weights false}}]
   (check-available)
   (let [ptr (ffi/with-c-string [p (str path)] (dt-model-load p (if gpu? 1 0) n-threads (if f32-weights 1 0)))]
     (when (ffi/null? ptr)
       (throw (ex-info (str "cannot load " path " for training: " (dt-error)) {:path (str path)})))
     (->GradModel ptr (str path) (dt-model-n-embd ptr) (dt-model-n-vocab ptr) (= 1 (dt-model-on-gpu ptr))))))

(defn loss-and-grad
  "The decoder's mean cross-entropy over the scored positions of one sequence,
  and its gradient with respect to the memory rows.
  tokens: token id per position (anything at memory positions);
  mem-index: memory row per position, or -1 for a token position;
  targets: the token each position must predict, or -1;
  memory: an n_mem x n_embd matrix. With :grad? false only the loss is computed.
  Returns {:loss :n-targets :grad (matrix or nil)}."
  [{:keys [ptr n-embd]} {:keys [tokens mem-index targets memory]} & {:keys [grad?] :or {grad? true}}]
  (let [n (count tokens)
        n-mem (:rows memory)]
    (when-not (= (:cols memory) n-embd)
      (throw (ex-info "memory width does not match the decoder" {:cols (:cols memory) :n-embd n-embd})))
    (with-open [arena (ffi/confined-arena)]
      (let [ints (fn [xs] (let [p (ffi/alloc arena (* 4 (max 1 (count xs))))]
                            (when (seq xs) (ffi/write-array p :int32 (int-array xs)))
                            p))
            tp (ints tokens) mp (ints mem-index) gp (ints targets)
            memp (ffi/alloc arena (* 4 (max 1 (* n-mem n-embd))))
            _ (when (pos? n-mem) (ffi/write-array memp :float (:data memory)))
            gradp (ffi/alloc arena (* 4 (max 1 (* n-mem n-embd))))
            lossp (ffi/alloc arena 4)
            rc (dt-step ptr tp mp gp n memp n-mem gradp lossp (if grad? 1 0))]
        (when (neg? rc) (throw (ex-info (str "training step failed: " (dt-error)) {})))
        {:loss (double (ffi/read lossp :float))
         :n-targets rc
         :grad (when grad? (t/matrix n-mem n-embd (if (pos? n-mem) (ffi/read-array gradp :float (* n-mem n-embd))
                                                     (float-array 0))))}))))
