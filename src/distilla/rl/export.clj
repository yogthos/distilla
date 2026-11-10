(ns distilla.rl.export
  "Export the memory module of an RL checkpoint as a Memorilla checkpoint
  (memorilla rl/export_memory.py).

  Reads distilla's own RL checkpoints (global_step_N/ holding the module) and
  single-GPU SkyRL-Memorilla checkpoints (global_step_N/policy/
  model_world_size_1_rank_0.pt, which also hold the frozen decoder: only the
  memory module's tensors are read). The architecture is inferred from the
  weights; the number of heads leaves no trace and comes from --num_heads."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [distilla.cli :as cli]
            [distilla.memory :as memory]
            [distilla.torch :as torch]
            [distilla.util :as util]))

(def policy-state-file ["policy" "model_world_size_1_rank_0.pt"])

(defn encoder-prefix [modality] (str "_skyrl_modality_encoders." modality ".memory."))

(defn extract-memory-state
  "The memory module's tensors of one modality from a SkyRL checkpoint."
  [checkpoint modality]
  (let [file (str (apply fs/path checkpoint policy-state-file))]
    (when-not (fs/exists? file)
      (throw (ex-info (str "Expected the policy state of a single-GPU run at " file ".") {})))
    (let [state (torch/load file {:lazy true})
          state (if (map? (get state "state_dict")) (get state "state_dict") state)
          prefix (encoder-prefix modality)
          picked (into {} (for [[k v] state
                                :when (and (string? k) (str/includes? k prefix) (torch/tensor? v))]
                            [(second (str/split k (re-pattern (java.util.regex.Pattern/quote prefix)) 2))
                             (torch/realize v)]))]
      (when (empty? picked)
        (throw (ex-info (str "No `" prefix "*` weights in " file ".") {})))
      picked)))

(defn export!
  "Write memory.safetensors and config.json for checkpoint to output-dir."
  [checkpoint output-dir {:keys [modality num-heads] :or {modality "memorilla" num-heads memory/default-num-heads}}]
  (let [own (util/checkpoint-file checkpoint)
        state (if (fs/exists? own)
                (memory/load-memory-state own)
                (extract-memory-state checkpoint modality))
        cfg (assoc (memory/config-from-state-dict state) :num-heads num-heads)]
    (with-open [m (memory/from-state cfg state)]
      (memory/save m output-dir)
      (println (str "Saved " (:config m) " to " output-dir)))))

(def spec
  {:modality {:desc "Modality name in the training config." :default "memorilla"}
   :num_heads {:desc "Attention heads of the memory module." :default memory/default-num-heads :coerce :long}})

(defn -main [& args]
  (cli/exit-on-error
   #(let [[checkpoint output-dir & flags] args]
      (when-not (and checkpoint output-dir)
        (throw (ex-info "usage: jolt rl-export CHECKPOINT OUTPUT_DIR [--num_heads 8] [--modality memorilla]" {})))
      (let [opts (cli/parse "usage: jolt rl-export CHECKPOINT OUTPUT_DIR" spec flags)]
        (export! checkpoint output-dir {:modality (:modality opts) :num-heads (:num_heads opts)}))))
  (System/exit 0))
