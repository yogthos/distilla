(ns distilla.answer
  "Answer one question from a collection of documents, the README's getting
  started example:

    jolt answer --checkpoint memorilla/Memorilla-Qwen3-8B --subfolder pmv2 \\
      --decoder Qwen3-8B-Q8_0.gguf --encoder Qwen3-Embedding-4B-Q8_0.gguf \\
      --question \"What are some easy breakfast ideas?\" --documents docs.json

  docs.json holds a JSON array of strings. --row N with --config/--split
  takes the question and documents from a dataset row instead."
  (:require [distilla.benchmarks :as b]
            [distilla.cli :as cli]
            [distilla.data :as data]
            [distilla.dataset :as ds]
            [distilla.model :as model]
            [distilla.paths :as paths]
            [distilla.util :as util]))

(def spec
  {:checkpoint {:desc "Checkpoint directory or Hub repo id." :required true}
   :subfolder {:desc "Checkpoint folder inside a Hub repo (e.g. pmv2)."}
   :decoder {:desc "Decoder GGUF." :required true}
   :encoder {:desc "Encoder GGUF." :required true}
   :question {:desc "The question."}
   :documents {:desc "JSON file with an array of document texts."}
   :config {:desc "Take the question and documents from this dataset config."}
   :split {:desc "Split for --config." :default "test"}
   :row {:desc "Row of --config/--split." :coerce :long}
   :system_prompt {:desc "System prompt." :default b/persona-prompt}
   :max_new_tokens {:desc "Generation budget." :default 256 :coerce :long}
   :max_model_len {:desc "Decoder context length." :default 8192 :coerce :long}
   :n_gpu_layers {:desc "Layers offloaded to the GPU (-1 = all)." :default -1 :coerce :long}
   :cache_dir {:desc "Download directory for Hub checkpoints." :default paths/model-cache-dir}})

(defn run [opts]
  (let [[question documents]
        (if (:config opts)
          (let [root (data/ensure-data (:config opts) [(:split opts)])
                row (ds/row (data/load-split (:config opts) (:split opts) root) (:row opts 0))]
            [(get row data/question-column) (get row data/documents-column)])
          [(or (:question opts) (throw (ex-info "pass --question (or --config and --row)" {})))
           (if (:documents opts) (util/read-json (:documents opts)) [])])]
    (with-open [m (model/from-pretrained (:checkpoint opts)
                                         {:subfolder (:subfolder opts) :decoder (:decoder opts)
                                          :encoder (:encoder opts) :max-model-len (:max_model_len opts)
                                          :n-gpu-layers (:n_gpu_layers opts) :cache-dir (:cache_dir opts)})]
      (let [answer (model/answer m question documents {:system-prompt (:system_prompt opts)
                                                       :max-new-tokens (:max_new_tokens opts)})]
        (println (str "Question: " question "\nAnswer: " answer))))))

(defn -main [& args]
  (cli/exit-on-error #(run (cli/parse "usage: jolt answer --checkpoint DIR --decoder Q.gguf --encoder E.gguf --question Q --documents docs.json" spec args))))
