(ns distilla.evaluate
  "Evaluate a trained memory module: generate with a frozen llama.cpp
  decoder, then score.

  Registered benchmarks follow their evaluation protocol (--benchmark
  <name|all>); any other dataset split can be evaluated with --config <name>
  --split <split> --scoring <scoring>."
  (:require [babashka.fs :as fs]
            [distilla.benchmarks :as b]
            [distilla.cli :as cli]
            [distilla.data :as data]
            [distilla.dataset :as ds]
            [distilla.embeddings :as emb]
            [distilla.memory :as memory]
            [distilla.metrics :as metrics]
            [distilla.model :as model]
            [distilla.paths :as paths]
            [distilla.util :as util]))

(def custom-max-length 2048)
(def custom-max-new-tokens 128)

(def spec
  {:benchmark {:desc "Registered benchmark name, or 'all'."}
   :config {:desc "Dataset config to evaluate instead of a registered benchmark."}
   :split {:desc "Split to evaluate." :default data/test-split}
   :scoring {:desc "Scoring method for --config (choice, exact_match, polarity, generation)." :default b/generation}
   :checkpoint {:desc "Checkpoint directory, weights file, or Hub repo id." :required true}
   :subfolder {:desc "Checkpoint folder inside a Hub repo (e.g. pv4)."}
   :decoder {:desc "Decoder GGUF the checkpoint was trained with (Qwen3-8B)." :required true}
   :encoder {:desc "Encoder GGUF (Qwen3-Embedding-4B), for choice scoring and embeddings that must be computed."}
   :encoder_id {:desc "Encoder id naming the stored embeddings." :default paths/default-encoder}
   :system_prompt {:desc "Override the system prompt."}
   :system_prompt_from {:desc "Use the system prompt a training recipe (YAML) trains with."}
   :max_length {:desc "Override the prompt token limit." :coerce :long}
   :max_new_tokens {:desc "Override the generation budget." :coerce :long}
   :batch_size {:desc "Rows per memory batch (padding as in the Python evaluation)." :default 32 :coerce :long}
   :n_gpu_layers {:desc "Layers offloaded to the GPU (-1 = all)." :default -1 :coerce :long}
   :data_repo {:desc "Hub dataset repo or local directory." :default paths/data-repo}
   :data_dir {:desc "Local mirror of a Hub data repo." :default paths/data-dir}
   :cache_dir {:desc "Download directory for Hub checkpoints." :default paths/model-cache-dir}
   :output_dir {:desc "Results directory (one folder per benchmark)." :default paths/results-dir}})

(defn resolve-targets
  "The benchmarks to evaluate, with overrides applied."
  [opts]
  (let [benchmarks (if (:benchmark opts)
                     (b/get-benchmarks (:benchmark opts))
                     [(b/benchmark {:name (:config opts) :config (:config opts) :split data/test-split
                                    :system-prompt data/default-system-prompt
                                    :max-length custom-max-length :max-new-tokens custom-max-new-tokens
                                    :scoring (:scoring opts)
                                    :metrics (b/metrics-by-scoring (:scoring opts))})])
        split (or (:split opts) data/test-split)
        benchmarks (if (not= split data/test-split)
                     (mapv #(assoc % :name (str (:name %) "-" split) :split split) benchmarks)
                     benchmarks)
        system-prompt (if (:system_prompt_from opts)
                        (get (util/read-recipe (:system_prompt_from opts)) "system_prompt" data/default-system-prompt)
                        (:system_prompt opts))
        overrides (cond-> {}
                    system-prompt (assoc :system-prompt system-prompt)
                    (:max_length opts) (assoc :max-length (:max_length opts))
                    (:max_new_tokens opts) (assoc :max-new-tokens (:max_new_tokens opts)))]
    (mapv #(merge % overrides) benchmarks)))

(defn- make-encoder-fn
  "A fn starting the encoder from --encoder (nil when none is given)."
  [opts]
  (when-let [path (:encoder opts)]
    (fn []
      (let [load-model (requiring-resolve 'distilla.llama/load-model)
            encoder (requiring-resolve 'distilla.llama/encoder)
            m (load-model path {:n-gpu-layers (:n_gpu_layers opts)})
            e (encoder m)]
        ;; the context is closed with the encoder; the model is left to the process exit
        e))))

(defn load-target
  "Fetch a benchmark's split and embeddings and check the split can be
  scored: [benchmark dataset store]."
  [benchmark opts]
  (let [root (data/ensure-data (:config benchmark) [(:split benchmark)] (:data_repo opts) (:data_dir opts))
        _ (emb/ensure-embeddings (:config benchmark) [(:split benchmark)]
                                 {:repo (:data_repo opts) :data-dir (:data_dir opts)
                                  :encoder-id (:encoder_id opts) :make-encoder (make-encoder-fn opts)})
        [dataset st] (emb/load-split-with-embeddings (:config benchmark) (:split benchmark) root
                                                     data/eval-columns (:encoder_id opts))
        benchmark (if (= (:scoring benchmark) b/choice)
                    (assoc benchmark :metrics (if (and (ds/has-column? dataset data/hard-column)
                                                       (some true? (ds/column dataset data/hard-column)))
                                                [b/accuracy b/accuracy-hard]
                                                [b/accuracy]))
                    benchmark)]
    (metrics/check-scoring-inputs benchmark dataset)
    [benchmark dataset st]))

(defn generate-split
  "One cleaned generation per row of a split."
  [m benchmark dataset st batch-size]
  (let [items (data/memory-dataset [[(:config benchmark) (:split benchmark) dataset]])
        n (data/dataset-size items)
        collate-opts {:tokenizer (:tokenizer m)
                      :stores {[(:config benchmark) (:split benchmark)] st}
                      :num-memories (memory/num-memories (:memory m))
                      :max-length (:max-length benchmark)
                      :system-prompt (:system-prompt benchmark)
                      :suffix (:suffix benchmark)
                      :train false}]
    (vec (mapcat (fn [start]
                   (let [batch (mapv #(data/dataset-item items %) (range start (min n (+ start batch-size))))
                         texts (model/generate m (data/collate collate-opts batch)
                                               {:max-new-tokens (:max-new-tokens benchmark)})]
                     (println (str (:name benchmark) ": " (min n (+ start batch-size)) "/" n))
                     (map metrics/clean-generation texts)))
                 (range 0 n batch-size)))))

(defn run
  "Generate and score every requested target."
  [opts]
  (when-not (or (:benchmark opts) (:config opts))
    (throw (ex-info "pass --benchmark or --config" {})))
  (let [targets (mapv #(load-target % opts) (resolve-targets opts))
        checkpoint (model/resolve-checkpoint (:checkpoint opts) (:subfolder opts) (:cache_dir opts))
        mem (memory/from-pretrained (paths/resolve-path checkpoint))
        load-model (requiring-resolve 'distilla.llama/load-model)
        make-decoder (requiring-resolve 'distilla.llama/decoder)
        make-tokenizer (requiring-resolve 'distilla.llama/tokenizer)
        _ ((requiring-resolve 'distilla.llama/quiet!) true)
        dec-model (load-model (:decoder opts) {:n-gpu-layers (:n_gpu_layers opts)})
        _ (when-not (= (:n-embd dec-model) (get-in mem [:config :output-dim]))
            (throw (ex-info (str "The checkpoint produces " (get-in mem [:config :output-dim])
                                 "-dimensional memory tokens but " (:decoder opts) " has hidden size "
                                 (:n-embd dec-model) "; pass the decoder the checkpoint was trained with.")
                            {})))
        dec (make-decoder dec-model {:n-ctx (apply max (map (comp :max-length first) targets))})
        m (model/memory-model {:memory mem :tokenizer (make-tokenizer dec-model)
                               :engine (model/llama-engine dec) :closeables [mem dec-model dec]})
        output-dir (paths/resolve-path (:output_dir opts))
        jobs (try
               (mapv (fn [[benchmark dataset st]]
                       [benchmark dataset (generate-split m benchmark dataset st (:batch_size opts))
                        (str (fs/path output-dir (:name benchmark)))])
                     targets)
               (finally (.close ^java.io.Closeable m)))
        encoder (atom nil)]
    (try
      (metrics/score-predictions jobs (fn []
                                        (let [make (or (make-encoder-fn opts)
                                                       (throw (ex-info "choice scoring needs --encoder" {})))]
                                          (reset! encoder (make)))))
      (finally (some-> ^java.io.Closeable @encoder .close)))))

(defn -main [& args]
  (cli/exit-on-error
   #(run (cli/parse "usage: jolt evaluate --benchmark <name|all> --checkpoint DIR --decoder Q.gguf [--encoder E.gguf]"
                    spec args))))
