(ns distilla.rl.main
  "jolt rl-train / jolt rl-eval: GRPO training of the memory module on
  TextWorld and its evaluation (memorilla rl/train.sh, rl/evaluate.sh), with
  the settings of rl/configs/textworld_{grpo,eval}.yaml as defaults
  (configs/rl/*.yaml here; any key can be given as a flag)."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [distilla.cli :as cli]
            [distilla.memory :as memory]
            [distilla.optim :as optim]
            [distilla.parquet :as pq]
            [distilla.rl.aggregate :as aggregate]
            [distilla.rl.grpo :as grpo]
            [distilla.tokenizer :as tok]
            [distilla.util :as util]))

(def spec
  {:config {:desc "YAML file of option defaults (configs/rl/textworld_grpo.yaml)."}
   :train_data {:desc "Training parquet (rl-dataset output)."}
   :val_data {:desc "Validation / evaluation parquet."}
   :decoder {:desc "Frozen agent GGUF (Qwen3-4B-Instruct-2507)."}
   :encoder {:desc "Frozen embedding GGUF (Qwen3-Embedding-4B)."}
   :template {:desc "Chat template of the agent: chatml (Instruct-2507) or qwen3." :default "chatml"}
   :num_memories {:desc "Memory vectors (K)." :default 8 :coerce :long}
   :num_heads {:default 8 :coerce :long}
   :num_self_attn_layers {:default 1 :coerce :long}
   :num_cross_attn_layers {:default 2 :coerce :long}
   :dropout {:default 0.1 :coerce :double}
   :checkpoint_path {:desc "Memory checkpoint to start from (training) or evaluate."}
   :model {:desc "rl-eval: a checkpoint, `untrained` or `no-memory`."}
   :max_turns {:default 50 :coerce :long}
   :inform7 {:desc "Compiled (.z8) games: bridge (TextWorld's Inform7) or simulator (their .json spec)." :default "bridge"}
   :python {:desc "Python with textworld, for the Inform7 bridge." :default "python3"}
   :n_samples_per_prompt {:default 16 :coerce :long}
   :eval_n_samples_per_prompt {:default 1 :coerce :long}
   :max_generate_length {:default 128 :coerce :long}
   :temperature {:default 1.0 :coerce :double}
   :eval_temperature {:desc "0 is greedy." :default 0.0 :coerce :double}
   :max_input_length {:default 16000 :coerce :long}
   :n_ctx {:desc "Agent context length." :default 16384 :coerce :long}
   :lr {:default 5e-4 :coerce :double}
   :weight_decay {:default 0.01 :coerce :double}
   :max_grad_norm {:default 1.0 :coerce :double}
   :epochs {:default 10 :coerce :long}
   :train_batch_size {:desc "Prompts per step." :default 4 :coerce :long}
   :micro_train_batch_size {:desc "Trajectories per loss micro-batch." :default 2 :coerce :long}
   :eval_interval {:default 3 :coerce :long}
   :ckpt_interval {:default 3 :coerce :long}
   :max_ckpts_to_keep {:desc "Keep the newest N checkpoints (0 keeps all)." :default 0 :coerce :long}
   :seed {:default 42 :coerce :long}
   :ckpt_path {:desc "Checkpoint directory (resumes from the latest)."}
   :export_path {:desc "Logs and evaluation dumps."}
   :n_gpu_layers {:default -1 :coerce :long}
   :f32_weights {:desc "Widen the training copy of the decoder to f32 (CUDA backward)." :default false}})

(defn- coerce [k v]
  (case (get-in spec [k :coerce])
    :long (if (string? v) (parse-long v) (long v))
    :double (if (string? v) (parse-double v) (double v))
    (if (#{:f32_weights} k) (if (string? v) (util/str->bool v) (boolean v)) v)))

(defn parse-args
  "Defaults, then the --config YAML, then flags. Unknown YAML keys are errors."
  [argv]
  (let [opts (cli/parse "usage: jolt rl-train|rl-eval --config FILE [--key value ...]"
                        (update-vals spec #(select-keys % [:desc])) argv)
        recipe (when (:config opts) (util/read-recipe (:config opts)))
        unknown (remove #(contains? spec (keyword %)) (keys recipe))]
    (when (seq unknown) (throw (ex-info (str (:config opts) " sets unknown options: " (vec (sort unknown))) {})))
    (let [defaults (into {} (for [[k {:keys [default]}] spec :when (some? default)] [k default]))
          merged (merge defaults (into {} (for [[k v] recipe] [(keyword k) v])) (into {} (remove (comp nil? val) opts)))]
      (into {} (for [[k v] merged] [k (if (some? v) (coerce k v) v)])))))

;; --- the policy ------------------------------------------------------------------------------------

(defn- llama [sym] (requiring-resolve (symbol "distilla.llama" (name sym))))

(defn load-policy
  "Agent, encoder and memory module. mode: :train, :checkpoint, :untrained or :no-memory."
  [opts mode]
  ((llama 'quiet!) true)
  (let [agent ((llama 'load-model) (:decoder opts) {:n-gpu-layers (:n_gpu_layers opts)})
        dec ((llama 'decoder) agent {:n-ctx (:n_ctx opts)})
        template (if (= "qwen3" (:template opts)) tok/qwen3 tok/chatml)
        tokenizer ((llama 'tokenizer) agent template)
        enc-model (when-not (= mode :no-memory) ((llama 'load-model) (:encoder opts) {:n-gpu-layers (:n_gpu_layers opts)}))
        enc (when enc-model ((llama 'encoder) enc-model))
        cfg (when enc-model
              (memory/config {:embedding-dim (:n-embd enc-model) :output-dim (:n-embd agent)
                              :num-memories (:num_memories opts) :num-heads (:num_heads opts)
                              :num-self-attn-layers (:num_self_attn_layers opts)
                              :num-cross-attn-layers (:num_cross_attn_layers opts) :dropout (:dropout opts)}))
        module (case mode
                 :no-memory nil
                 :checkpoint (memory/from-pretrained (:checkpoint_path opts) {:num-heads (:num_heads opts)
                                                                               :dropout (:dropout opts)})
                 (let [m (grpo/init-memory-module cfg (:seed opts))]
                   (when-let [init (and (= mode :train) (:checkpoint_path opts))]
                     (memory/load-partial! m (memory/load-memory-state init)))
                   m))]
    {:generate (fn [segments o] ((llama 'generate) dec segments (assoc o :tokens? true)))
     :tokenizer tokenizer
     :eos (:eos agent)
     :memory-module module
     :encoder enc
     :num-memories (:num_memories opts)
     :rng (memory/rng (:seed opts))
     :closeables (filterv some? [module enc enc-model dec agent])}))

(defn close-policy [policy] (doseq [c (:closeables policy)] (.close ^java.io.Closeable c)))

(defn- sampling [opts eval?]
  {:max-turns (:max_turns opts) :max-input-length (:max_input_length opts) :max-new-tokens (:max_generate_length opts)
   :temperature (if eval? (:eval_temperature opts) (:temperature opts))
   :inform7 (:inform7 opts) :python (:python opts)})

;; --- checkpoints -------------------------------------------------------------------------------------

(defn- step-dirs [ckpt-path]
  (->> (when (fs/exists? ckpt-path) (fs/glob ckpt-path "global_step_*"))
       (keep (fn [d] (when-let [[_ n] (re-matches #"global_step_(\d+)" (str (fs/file-name d)))] [(parse-long n) (str d)])))
       (sort-by first)))

(defn save-checkpoint! [{:keys [memory-module optimizer]} ckpt-path state keep]
  (let [dir (str (fs/path ckpt-path (str "global_step_" (:step state))))]
    (memory/save memory-module dir)
    (optim/save-state optimizer (str (fs/path dir "optimizer.safetensors")))
    (util/write-json (fs/path dir "trainer_state.json") {"step" (:step state) "epoch" (:epoch state) "batch" (:batch state)})
    (when (pos? keep)
      (doseq [[_ d] (drop-last keep (step-dirs ckpt-path))] (fs/delete-tree d)))
    dir))

(defn- resume! [{:keys [memory-module optimizer]} ckpt-path]
  (when-let [[_ dir] (last (step-dirs ckpt-path))]
    (memory/load-partial! memory-module (memory/load-memory-state dir))
    (optim/load-state! optimizer (str (fs/path dir "optimizer.safetensors")))
    (let [s (util/read-json (fs/path dir "trainer_state.json"))]
      (println "resumed from" dir)
      {:step (get s "step") :epoch (get s "epoch") :batch (get s "batch")})))

;; --- commands -----------------------------------------------------------------------------------------

(defn- eval-dump! [policy opts rows step]
  (let [dir (str (fs/path (:export_path opts) "dumped_evals" (str "global_step_" step "_evals")))]
    (grpo/evaluate! policy rows {:n-samples (:eval_n_samples_per_prompt opts) :sampling (sampling opts true)
                                 :dump-dir dir})
    (print (aggregate/table (aggregate/average [(aggregate/score-rollouts (aggregate/load-rollouts dir))])))
    dir))

(defn train [opts]
  (doseq [k [:train_data :decoder :encoder :ckpt_path :export_path]]
    (when-not (get opts k) (throw (ex-info (str "--" (name k) " is required") {}))))
  (let [policy (load-policy opts :train)
        optimizer (optim/adamw (:memory-module policy) {:weight-decay (:weight_decay opts)})
        policy (assoc policy :optimizer optimizer)
        _ (memory/ensure-grads! (:memory-module policy))
        grad ((llama 'grad-model) (:decoder opts) {:f32-weights (:f32_weights opts)})
        decoder (fn [sequence grad?] ((llama 'loss-and-grad) grad sequence :grad? grad?))
        rows (pq/read-rows [(:train_data opts)])
        val-rows (when (:val_data opts) (pq/read-rows [(:val_data opts)]))
        start (or (resume! policy (:ckpt_path opts)) {:step 0 :epoch 0 :batch 0})
        batch-size (:train_batch_size opts)
        log (str (fs/path (:export_path opts) "metrics.jsonl"))]
    (fs/create-dirs (:export_path opts))
    (try
      (doseq [epoch (range (:epoch start) (:epochs opts))]
        (let [rng (memory/rng (+ (:seed opts) epoch))
              order (sort-by (fn [_] (memory/uniform! rng)) (range (count rows)))
              batches (vec (partition-all batch-size order))]
          (doseq [b (range (if (= epoch (:epoch start)) (:batch start) 0) (count batches))]
            (let [result (grpo/train-step! policy decoder (mapv rows (nth batches b))
                                           {:n-samples (:n_samples_per_prompt opts)
                                            :micro-batch (:micro_train_batch_size opts)
                                            :lr (:lr opts) :max-grad-norm (:max_grad_norm opts)
                                            :sampling (sampling opts false)})
                  step (:step result)
                  row (-> (dissoc result :trajectories) (assoc :epoch epoch))]
              (spit log (str (clojure.data.json/write-str row) "\n") :append true)
              (println (format "step %d epoch %d return %.3f solve %.3f loss %.4f grad %.3f"
                               step epoch (:mean-return result) (:solve-rate result) (:policy-loss result)
                               (:grad-norm result)))
              (when (and val-rows (pos? (:eval_interval opts)) (zero? (mod step (:eval_interval opts))))
                (eval-dump! policy opts val-rows step))
              (when (and (pos? (:ckpt_interval opts)) (zero? (mod step (:ckpt_interval opts))))
                (println "saved" (save-checkpoint! policy (:ckpt_path opts)
                                                   {:step step :epoch epoch :batch (inc b)}
                                                   (:max_ckpts_to_keep opts))))))))
      (let [step @(:step optimizer)]
        (save-checkpoint! policy (:ckpt_path opts) {:step step :epoch (:epochs opts) :batch 0} (:max_ckpts_to_keep opts)))
      (finally
        (.close ^java.io.Closeable grad)
        (close-policy policy)))))

(defn evaluate [opts]
  (doseq [k [:val_data :decoder :export_path]]
    (when-not (get opts k) (throw (ex-info (str "--" (name k) " is required") {}))))
  (let [model (or (:model opts) (:checkpoint_path opts) "untrained")
        mode (case model "untrained" :untrained "no-memory" :no-memory :checkpoint)
        opts (cond-> opts (= mode :checkpoint) (assoc :checkpoint_path model))
        policy (load-policy opts mode)
        rows (pq/read-rows [(:val_data opts)])]
    (try
      (let [dir (str (fs/path (:export_path opts) "dumped_evals" "eval_evals"))]
        (grpo/evaluate! policy rows {:n-samples (:eval_n_samples_per_prompt opts) :sampling (sampling opts true)
                                     :dump-dir dir})
        (aggregate/run [(:export_path opts)] {:output (str (fs/path (:export_path opts) "metrics.json"))}))
      (finally (close-policy policy)))))

(defn -main [command & args]
  (cli/exit-on-error
   #(case command
      "train" (train (parse-args args))
      "eval" (evaluate (parse-args args))
      (throw (ex-info (str "unknown command " command " (train or eval)") {}))))
  (System/exit 0))
