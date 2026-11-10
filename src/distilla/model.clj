(ns distilla.model
  "Generation from a frozen decoder conditioned on memory tokens: the
  counterpart of the Python MemoryVLLM.

  The memory module turns document and question embeddings into memory
  tokens; each prompt is cut at its <|memory|> placeholders into segments of
  token ids and memory vectors, which the decoder reads in order. That is the
  same computation as overwriting the placeholder embeddings."
  (:require [babashka.fs :as fs]
            [distilla.data :as data]
            [distilla.encoder :as encoder]
            [distilla.hub :as hub]
            [distilla.memory :as memory]
            [distilla.paths :as paths]
            [distilla.tensor :as t]
            [distilla.tokenizer :as tok]))

(defprotocol Engine
  (generate-text [engine segments opts]
    "Generate from prompt segments ([:tokens ids] / [:embeddings matrix]). opts:
    :max-new-tokens :temperature :top-p :seed. Returns the text."))

(defn prompt-segments
  "Cut a prompt's ids at the memory placeholders: runs of token ids become
  [:tokens ids], each run of placeholders [:embeddings rows] taking the next
  rows of the memory matrix in order. Every memory row must be used once."
  [ids memory-id memory-matrix]
  (let [runs (partition-by #(= % memory-id) ids)
        [segments used] (reduce (fn [[segs used] run]
                                  (if (= (first run) memory-id)
                                    (let [n (count run)]
                                      [(conj segs [:embeddings (t/select-rows memory-matrix (range used (+ used n)))])
                                       (+ used n)])
                                    [(conj segs [:tokens (vec run)]) used]))
                                [[] 0] runs)]
    (when-not (= used (:rows memory-matrix))
      (throw (ex-info (str "the prompt has " used " memory placeholders for " (:rows memory-matrix) " memory tokens")
                      {:placeholders used :memories (:rows memory-matrix)})))
    segments))

(defrecord MemoryModel [memory tokenizer engine encoder closeables]
  java.io.Closeable
  (close [_] (doseq [c (reverse closeables)] (.close ^java.io.Closeable c))))

(defn memory-model
  "Assemble a model. engine: an Engine (e.g. a llama decoder via
  llama-engine); encoder: an Encoder, needed only by answer; closeables are
  closed with the model."
  [{:keys [memory tokenizer engine encoder closeables]}]
  (->MemoryModel memory tokenizer engine encoder (vec closeables)))

(defn generate
  "One completion per row of a collated batch (distilla.data/collate in
  generation mode). opts: :max-new-tokens (256) :temperature (0, greedy)
  :top-p (1.0) :seed."
  ([model batch] (generate model batch {}))
  ([{:keys [memory tokenizer engine]} {:keys [input-ids attention-mask doc-embeds doc-padding-mask question-embeds]} opts]
   (let [memories (memory/forward-batch memory doc-embeds doc-padding-mask question-embeds)
         mem-id (tok/memory-token-id tokenizer)]
     (mapv (fn [ids mask mem]
             (let [ids (vec (keep-indexed (fn [i id] (when (pos? (nth mask i)) id)) ids))]
               (generate-text engine (prompt-segments ids mem-id mem)
                              (merge {:max-new-tokens 256 :temperature 0.0 :top-p 1.0} opts))))
           input-ids attention-mask memories))))

(defn answer
  "Answer one question from document texts: the encoder embeds every document
  and the question, the memory module compresses them, and the decoder
  answers greedily with the memory tokens in its system prompt."
  ([model question documents] (answer model question documents {}))
  ([{:keys [memory tokenizer encoder] :as model} question documents
    {:keys [system-prompt max-new-tokens] :or {system-prompt data/default-system-prompt max-new-tokens 256}}]
   (when-not encoder
     (throw (ex-info "answer needs an encoder; load the model with from-pretrained or pass :encoder." {})))
   (let [vectors (encoder/embed-texts encoder (concat documents [question]))
         n (count documents)
         messages (data/build-messages question system-prompt (memory/num-memories memory))
         prompt (tok/apply-chat-template tokenizer messages true)
         ids (tok/encode tokenizer prompt)
         [text] (generate model
                          {:input-ids [ids]
                           :attention-mask [(vec (repeat (count ids) 1))]
                           :doc-embeds [(t/take-rows vectors n)]
                           :doc-padding-mask [(vec (repeat n false))]
                           :question-embeds [(t/row vectors n)]}
                          {:max-new-tokens max-new-tokens})]
     (clojure.string/trim text))))

;; --- loading ----------------------------------------------------------------------------

(defn resolve-checkpoint
  "A local checkpoint directory: checkpoint itself when it exists, otherwise
  a Hub model repo (e.g. memorilla/Memorilla-Qwen3-8B) of which only
  subfolder is downloaded into cache-dir."
  [checkpoint subfolder cache-dir]
  (let [path (paths/expand-home checkpoint)]
    (if (fs/exists? path)
      (str (if subfolder (fs/path path subfolder) path))
      (let [local (str (fs/path (or cache-dir (fs/path (System/getProperty "user.home") ".cache" "distilla"))
                                (str "models--" (clojure.string/replace checkpoint "/" "--"))))]
        (hub/snapshot-download {:repo checkpoint :repo-type :model :local-dir local
                                :patterns [(if subfolder (str subfolder "/*") "*")]})
        (str (if subfolder (fs/path local subfolder) local))))))

(defn llama-engine
  "An Engine over a distilla.llama decoder."
  [decoder]
  (let [generate (requiring-resolve 'distilla.llama/generate)]
    (reify Engine
      (generate-text [_ segments opts] (generate decoder segments opts)))))

(defn from-pretrained
  "Load a memory checkpoint with its decoder and encoder GGUF files.
  opts: :subfolder (e.g. \"pmv2\"), :decoder and :encoder (GGUF paths),
  :max-model-len (8192), :n-gpu-layers (-1), :cache-dir, :template (chat
  template fn, default Qwen3 with thinking off)."
  [checkpoint {:keys [subfolder decoder encoder max-model-len n-gpu-layers cache-dir template]
               :or {max-model-len 8192 n-gpu-layers -1}}]
  (let [load-model (requiring-resolve 'distilla.llama/load-model)
        make-decoder (requiring-resolve 'distilla.llama/decoder)
        make-encoder (requiring-resolve 'distilla.llama/encoder)
        make-tokenizer (requiring-resolve 'distilla.llama/tokenizer)
        quiet! (requiring-resolve 'distilla.llama/quiet!)
        _ (quiet! true)
        mem (memory/from-pretrained (resolve-checkpoint checkpoint subfolder cache-dir))
        dec-model (load-model decoder {:n-gpu-layers n-gpu-layers})
        _ (when-not (= (:n-embd dec-model) (get-in mem [:config :output-dim]))
            (throw (ex-info (str "The checkpoint produces " (get-in mem [:config :output-dim])
                                 "-dimensional memory tokens but " decoder " has hidden size " (:n-embd dec-model)
                                 "; pass the decoder the checkpoint was trained with.")
                            {})))
        dec (make-decoder dec-model {:n-ctx max-model-len})
        enc-model (when encoder (load-model encoder {:n-gpu-layers n-gpu-layers}))
        enc (when enc-model (make-encoder enc-model))]
    (when (and enc-model (not= (:n-embd enc-model) (get-in mem [:config :embedding-dim])))
      (throw (ex-info (str "The checkpoint reads " (get-in mem [:config :embedding-dim]) "-dimensional embeddings but "
                           encoder " produces " (:n-embd enc-model) ".")
                      {})))
    (memory-model {:memory mem
                   :tokenizer (if template (make-tokenizer dec-model template) (make-tokenizer dec-model))
                   :engine (llama-engine dec)
                   :encoder enc
                   :closeables (filterv some? [mem dec-model dec enc-model enc])})))
