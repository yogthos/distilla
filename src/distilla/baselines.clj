(ns distilla.baselines
  "The text-only baselines: closed book, retrieval (RAG-k) and full context.

  Documents go into the user turn as text (most similar first for RAG, stored
  order for full context), then the question and suffix; over-long prompts are
  truncated from the front. Outputs and scoring match distilla.evaluate."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [distilla.benchmarks :as b]
            [distilla.cli :as cli]
            [distilla.data :as data]
            [distilla.dataset :as ds]
            [distilla.embeddings :as emb]
            [distilla.metrics :as metrics]
            [distilla.paths :as paths]
            [distilla.store :as store]
            [distilla.tensor :as t]
            [distilla.tokenizer :as tok]
            [distilla.util :as util]))

(def closed-book "closed_book")
(def rag "rag")
(def full-context "full_context")
(def methods [closed-book rag full-context])
(def generation-reserve 32)
(def document-separator "\n\n")
(def context-file "context.json")
(def percentiles [50 90 95 99])

(def spec
  {:benchmark {:desc "Registered benchmark name, or 'all'." :required true}
   :method {:desc "Baseline to run: closed_book, rag or full_context." :required true}
   :top_k {:desc "Documents retrieved per question with --method rag." :default 5 :coerce :long}
   :decoder {:desc "Decoder GGUF (Qwen3-8B)." :required true}
   :encoder {:desc "Encoder GGUF (Qwen3-Embedding-4B), for choice scoring and embeddings that must be computed."}
   :encoder_id {:desc "Encoder id naming the stored embeddings." :default paths/default-encoder}
   :n_gpu_layers {:desc "Layers offloaded to the GPU (-1 = all)." :default -1 :coerce :long}
   :data_repo {:desc "Hub dataset repo or local directory." :default paths/data-repo}
   :data_dir {:desc "Local mirror of a Hub data repo." :default paths/data-dir}
   :output_dir {:desc "Results directory (one folder per method and benchmark)."
                :default (str (fs/path paths/results-dir "baselines"))}})

(defn method-label [method top-k]
  (if (= method rag) (str rag "_top" top-k) method))

(defn context-window
  "[max-model-len yarn-factor] of the decoder for a benchmark and method."
  [benchmark method]
  (if (= method full-context)
    [(get-in benchmark [:baseline :full-context-length]) (get-in benchmark [:baseline :yarn-factor])]
    [b/retrieval-context-length nil]))

(defn retrieve
  "Each row's top-k documents by cosine similarity to its question, most similar first."
  [dataset st top-k]
  (vec (map-indexed
        (fn [row item]
          (let [docs (get item data/documents-column)
                m (store/documents st (get item data/collection-column))
                normed (t/l2-normalize-rows (:data m) (:cols m))
                q (t/l2-normalize-rows (store/question st row) (:cols m))
                sims (vec (for [i (range (:rows m))]
                            (t/dot q (java.util.Arrays/copyOfRange ^floats normed (int (* i (:cols m)))
                                                                   (int (* (inc i) (:cols m)))))))
                order (take (min top-k (count docs)) (sort-by (fn [i] [(- (nth sims i)) i]) (range (count sims))))]
            (mapv #(nth docs %) order)))
        (ds/rows dataset))))

(defn user-message [documents question suffix]
  (if (seq documents)
    (str (str/join document-separator documents) document-separator question suffix)
    (str question suffix)))

(defn build-prompts
  "Tokenised chat prompts truncated from the front to max-input-tokens:
  [prompts untruncated-lengths]."
  [tokenizer benchmark dataset documents max-input-tokens]
  (let [encoded (mapv (fn [question docs]
                        (tok/encode tokenizer
                                    (tok/apply-chat-template
                                     tokenizer
                                     [{:role "system" :content (get-in benchmark [:baseline :system-prompt])}
                                      {:role "user" :content (user-message docs (str question)
                                                                           (get-in benchmark [:baseline :suffix]))}]
                                     true)))
                      (ds/column dataset data/question-column) documents)]
    [(mapv #(if (> (count %) max-input-tokens) (subvec % (- (count %) max-input-tokens)) %) encoded)
     (mapv count encoded)]))

(defn- percentile
  "numpy's default (linear) percentile."
  [sorted q]
  (let [n (count sorted)
        pos (* (/ q 100.0) (dec n))
        lo (int (Math/floor pos))
        hi (min (dec n) (inc lo))
        frac (- pos lo)]
    (+ (nth sorted lo) (* frac (- (nth sorted hi) (nth sorted lo))))))

(defn context-summary
  "How much of the intended context reached the decoder."
  [lengths max-model-len max-input-tokens]
  (let [n (count lengths)
        truncated (count (filter #(> % max-input-tokens) lengths))
        sorted (vec (sort lengths))]
    (apply array-map
           (concat
            ["max_model_len" max-model-len
             "max_input_tokens" max-input-tokens
             "num_prompts" n
             "num_truncated" truncated
             "fit_rate" (util/round2 (* 100.0 (/ (- n truncated) (double n))))
             "mean_prompt_tokens" (/ (Math/round (* 10.0 (/ (reduce + 0 lengths) (double n)))) 10.0)
             "max_prompt_tokens" (peek sorted)
             "tokens_dropped" (reduce + 0 (map #(max 0 (- % max-input-tokens)) lengths))]
            (mapcat (fn [q] [(str "p" q "_prompt_tokens") (double (percentile sorted q))]) percentiles)))))

(defn prepare-prompts
  "Load a benchmark's split, select its documents and tokenise the prompts."
  [benchmark opts tokenizer make-encoder]
  (let [method (:method opts)
        root (data/ensure-data (:config benchmark) [(:split benchmark)] (:data_repo opts) (:data_dir opts))
        columns (if (= method closed-book) data/eval-columns (conj data/eval-columns data/documents-column))
        [dataset documents]
        (cond
          (= method rag)
          (do (emb/ensure-embeddings (:config benchmark) [(:split benchmark)]
                                     {:repo (:data_repo opts) :data-dir (:data_dir opts)
                                      :encoder-id (:encoder_id opts) :make-encoder make-encoder})
              (let [[dataset st] (emb/load-split-with-embeddings (:config benchmark) (:split benchmark) root
                                                                 columns (:encoder_id opts))]
                [dataset (retrieve dataset st (:top_k opts))]))
          (= method full-context)
          (let [dataset (data/load-split (:config benchmark) (:split benchmark) root columns)]
            [dataset (mapv vec (ds/column dataset data/documents-column))])
          :else
          (let [dataset (data/load-split (:config benchmark) (:split benchmark) root columns)]
            [dataset (vec (repeat (ds/row-count dataset) []))]))
        _ (metrics/check-scoring-inputs benchmark dataset)
        [max-model-len yarn-factor] (context-window benchmark method)
        max-input-tokens (max (- max-model-len (get-in benchmark [:baseline :max-new-tokens]) generation-reserve) 1)
        [prompts lengths] (build-prompts tokenizer benchmark dataset documents max-input-tokens)]
    {:benchmark benchmark :dataset dataset :prompts prompts
     :context (context-summary lengths max-model-len max-input-tokens)
     :max-model-len max-model-len :yarn-factor yarn-factor}))

(defn run [opts]
  (when-not (some #{(:method opts)} methods)
    (throw (ex-info (str "--method must be one of " (str/join ", " methods)) {})))
  (let [load-model (requiring-resolve 'distilla.llama/load-model)
        make-decoder (requiring-resolve 'distilla.llama/decoder)
        make-tokenizer (requiring-resolve 'distilla.llama/tokenizer)
        generate (requiring-resolve 'distilla.llama/generate)
        make-llama-encoder (requiring-resolve 'distilla.llama/encoder)
        _ ((requiring-resolve 'distilla.llama/quiet!) true)
        make-encoder (when (:encoder opts)
                       #(make-llama-encoder (load-model (:encoder opts) {:n-gpu-layers (:n_gpu_layers opts)})))
        dec-model (load-model (:decoder opts) {:n-gpu-layers (:n_gpu_layers opts)})
        tokenizer (make-tokenizer dec-model)
        output-dir (fs/path (paths/resolve-path (:output_dir opts)) (method-label (:method opts) (:top_k opts)))
        prompt-sets (mapv #(prepare-prompts % opts tokenizer make-encoder) (b/get-benchmarks (:benchmark opts)))
        windows (group-by (juxt :max-model-len :yarn-factor) prompt-sets)
        jobs (vec (mapcat
                   (fn [[[max-model-len yarn-factor] sets]]
                     (with-open [dec (make-decoder dec-model (cond-> {:n-ctx max-model-len}
                                                               yarn-factor (assoc :yarn-factor yarn-factor
                                                                                  :yarn-orig-ctx b/yarn-original-max-position-embeddings)))]
                       (mapv (fn [{:keys [benchmark dataset prompts context]}]
                               (let [n (count prompts)
                                     predictions (vec (map-indexed
                                                       (fn [i ids]
                                                         (when (zero? (mod (inc i) 32))
                                                           (println (str (:name benchmark) ": " (inc i) "/" n)))
                                                         (metrics/clean-generation
                                                          (generate dec [[:tokens ids]]
                                                                    {:max-new-tokens (get-in benchmark [:baseline :max-new-tokens])})))
                                                       prompts))
                                     out (str (fs/path output-dir (:name benchmark)))]
                                 (util/write-json (fs/path out context-file) context)
                                 [benchmark dataset predictions out]))
                             sets)))
                   windows))
        _ (.close ^java.io.Closeable dec-model)
        encoder (atom nil)]
    (try
      (metrics/score-predictions jobs (fn []
                                        (when-not make-encoder (throw (ex-info "choice scoring needs --encoder" {})))
                                        (reset! encoder (make-encoder))))
      (finally (some-> ^java.io.Closeable @encoder .close)))))

(defn -main [& args]
  (cli/exit-on-error
   #(run (cli/parse "usage: jolt baselines --method <closed_book|rag|full_context> --benchmark <name|all> --decoder Q.gguf [--encoder E.gguf]"
                    spec args))))
