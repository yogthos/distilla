(ns distilla.rl.aggregate
  "Score TextWorld evaluation rollouts per difficulty tier (memorilla
  rl/aggregate.py). Reads the JSONL dumps of distilla.rl.grpo/evaluate! and
  of SkyRL (<eval_dir>/dumped_evals/<step>_evals/<tier>.jsonl alike).

  A rollout is solved when its final turn earns a positive reward. Per tier
  and overall: avg@k (mean solve rate over rollouts), pass@k (episodes solved
  by at least one rollout), mean total reward, the fraction of terminal
  rollouts (ended by the game or the turn limit) and of rollouts cut short by
  the input or generation budget. Several directories (e.g. seeds) are averaged."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [distilla.cli :as cli]
            [distilla.util :as util]))

(def summary-file "aggregated_results.jsonl")
(def overall "all")

(defn load-rollouts [eval-dir]
  (let [dirs (if (seq (fs/glob eval-dir "*.jsonl"))
               [(str eval-dir)]
               (vec (sort (distinct (map (comp str fs/parent) (fs/glob eval-dir "dumped_evals/*/*.jsonl"))))))]
    (when (> (count dirs) 1)
      (throw (ex-info (str eval-dir " holds several evaluations (" (str/join ", " (map fs/file-name dirs))
                           "); pass one of their directories instead.") {})))
    (let [files (for [d dirs f (sort (map str (fs/glob d "*.jsonl"))) :when (not= summary-file (str (fs/file-name f)))] f)]
      (when (empty? files) (throw (ex-info (str "No dumped evaluation rollouts under " eval-dir ".") {})))
      (vec (mapcat util/read-jsonl files)))))

(defn- final-reward [r] (let [s (get r "score")] (double (if (sequential? s) (peek (vec s)) s))))
(defn- total-reward [r] (let [s (get r "score")] (double (if (sequential? s) (reduce + 0.0 s) s))))

(defn score-rollouts [rollouts]
  (let [groups (reduce (fn [g r]
                         (let [extras (get r "env_extras")
                               episode [(get extras "game_file") (get-in extras ["extra_info" "row_index"])]]
                           (-> g
                               (update-in [(get r "data_source") episode] (fnil conj []) r)
                               (update-in [overall episode] (fnil conj []) r))))
                       {} rollouts)
        frac (fn [xs] (util/mean (map #(if % 1.0 0.0) xs)))]
    (into {} (for [[tier episodes] groups]
               (let [runs (vals episodes)
                     all (apply concat runs)]
                 [tier {"episodes" (count episodes)
                        "rollouts" (count all)
                        "avg@k" (frac (map #(pos? (final-reward %)) all))
                        "pass@k" (frac (map (fn [rs] (boolean (some #(pos? (final-reward %)) rs))) runs))
                        "mean_reward" (util/mean (map total-reward all))
                        "terminal" (frac (map #(= "stop" (get % "stop_reason")) all))
                        "truncated" (frac (map #(= "length" (get % "stop_reason")) all))}])))))

(def metric-names ["episodes" "rollouts" "avg@k" "pass@k" "mean_reward" "terminal" "truncated"])

(defn average
  "Mean of every metric over the runs that contain each tier, overall last."
  [runs]
  (let [tiers (sort-by (fn [t] [(= t overall) t]) (distinct (mapcat keys runs)))]
    (into (array-map)
          (for [tier tiers]
            [tier (into (array-map) (for [m metric-names]
                                      [m (util/mean (for [r runs :when (contains? r tier)] (get-in r [tier m])))]))]))))

(defn table [metrics]
  (str (format "%-10s %8s %7s %7s %7s %8s %9s\n" "tier" "episodes" "avg@k" "pass@k" "reward" "terminal" "truncated")
       (apply str (for [[tier v] metrics]
                    (format "%-10s %8.0f %7.3f %7.3f %7.2f %8.3f %9.3f\n" tier (double (v "episodes")) (v "avg@k")
                            (v "pass@k") (v "mean_reward") (v "terminal") (v "truncated"))))))

(def spec {:output {:desc "Optional JSON file for the metrics."}})

(defn run [eval-dirs opts]
  (let [metrics (average (mapv (comp score-rollouts load-rollouts) eval-dirs))]
    (print (table metrics))
    (when-let [out (:output opts)]
      (util/write-json out {"runs" (vec eval-dirs) "metrics" metrics}))
    metrics))

(defn -main [& args]
  (cli/exit-on-error
   #(let [dirs (vec (take-while (fn [a] (not (str/starts-with? a "--"))) args))
          opts (cli/parse "usage: jolt rl-aggregate EVAL_DIR... [--output metrics.json]" spec (drop (count dirs) args))]
      (when (empty? dirs) (throw (ex-info "pass at least one evaluation directory" {})))
      (run dirs opts)))
  (System/exit 0))
