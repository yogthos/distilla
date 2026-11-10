(ns distilla.rl.dataset
  "Generate TextWorld games for the difficulty tiers and build the RL dataset
  (memorilla rl/generate_games.py and rl/build_dataset.py).

  Game parameters come from the same seeded Python random stream (ported
  bit-exactly), so a run names and parameterises the same games. Building
  the games themselves is TextWorld's: compiled tiers through `tw-make`,
  spec tiers through textworld.generator.make_game (tools/tw_make_spec.py)."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [distilla.cli :as cli]
            [distilla.parquet :as pq]
            [distilla.pyrandom :as pr]
            [distilla.util :as util]
            [jolt.process :as p]))

;; --- tiers and the parameter stream --------------------------------------------------------------

(def game-seed-stride 9973)
(def max-quest-breadth 3)
(def attempts-per-game 4)

(def tiers
  (array-map
   "medium" {:world-size [15 25] :nb-objects [12 22] :quest-length [6 10] :compiled true}
   "hard" {:world-size [24 40] :nb-objects [20 35] :quest-length [10 16] :compiled true}
   "long" {:world-size [25 35] :nb-objects [30 50] :quest-length [15 22] :compiled false}
   "mega" {:world-size [35 50] :nb-objects [50 90] :quest-length [25 35] :compiled false}
   "huge" {:world-size [38 50] :nb-objects [60 100] :quest-length [40 55] :compiled false}
   "extreme" {:world-size [50 65] :nb-objects [80 130] :quest-length [60 75] :compiled false}))

(defn parse-mix
  "A tier mix such as medium:0.3,hard:0.7 as normalised weights, in order."
  [text]
  (let [pairs (for [item (str/split text #",")]
                (let [[tier weight] (str/split item #":")
                      tier (str/trim tier)
                      w (parse-double (str/trim weight))]
                  (when-not (contains? tiers tier)
                    (throw (ex-info (str "Unknown tier `" tier "`; choose from " (vec (sort (keys tiers))) ".") {})))
                  (when-not (pos? w) (throw (ex-info (str "Tier `" tier "` needs a positive weight.") {})))
                  [tier w]))
        total (reduce + (map second pairs))]
    (into (array-map) (for [[t w] pairs] [t (/ w total)]))))

(defn- choose-tier [rng mix]
  (let [threshold (pr/random rng)
        ordered (filter mix (keys tiers))]
    (loop [ts ordered cumulative 0.0]
      (let [t (first ts) c (+ cumulative (mix t))]
        (if (or (<= threshold c) (empty? (rest ts))) t (recur (rest ts) c))))))

(defn game-specs
  "The lazy stream of game specs (tier, parameters, seed, path) for indices 0, 1, 2, ..."
  [mix seed output-dir]
  (let [rng (pr/random-state seed)]
    (map (fn [index]
           (let [tier (choose-tier rng mix)
                 {:keys [world-size nb-objects quest-length compiled]} (tiers tier)
                 w (apply pr/randint rng world-size)
                 o (apply pr/randint rng nb-objects)
                 q (apply pr/randint rng quest-length)
                 breadth (if compiled 1 (pr/randint rng 1 (max 1 (min max-quest-breadth (quot q 4)))))
                 game-seed (+ seed (* game-seed-stride (inc index)))
                 name (format "game_%05d_%s_w%d_o%d_q%d_s%d" index tier w o q game-seed)]
             {:index index :tier tier :world-size w :nb-objects o :quest-length q :quest-breadth breadth
              :seed game-seed :path (str (fs/path output-dir (str name (if compiled ".z8" ".json"))))}))
         (range))))

(defn build-game!
  "Generate one game unless its file exists: nil on success, else the error."
  [{:keys [tier world-size nb-objects quest-length quest-breadth seed path]} {:keys [python tw-make]}]
  (when-not (fs/exists? path)
    (let [compiled (:compiled (tiers tier))
          {:keys [exit err]} (if compiled
                               (p/sh tw-make "custom" "--world-size" (str world-size) "--nb-objects" (str nb-objects)
                                     "--quest-length" (str quest-length) "--seed" (str seed) "--output" path)
                               (p/sh python (str (fs/path "tools" "tw_make_spec.py")) path (str seed) (str world-size)
                                     (str nb-objects) (str quest-length) (str quest-breadth)))]
      (when-not (zero? exit)
        (doseq [f (cons path (map #(str (fs/strip-ext path) %) [".json" ".ni"]))]
          (fs/delete-if-exists f))
        (str/trim (str err))))))

(def games-spec
  {:output_dir {:desc "Directory for the generated games." :required true}
   :tiers {:desc "Tier mix, e.g. medium:0.3,hard:0.7 or long:1." :required true}
   :num_games {:desc "Number of games to produce." :required true :coerce :long}
   :seed {:desc "Seed of the parameter stream." :default 42 :coerce :long}
   :workers {:desc "Games generated at once." :default 1 :coerce :long}
   :max_attempts {:desc "Game indices to try (default 4 per requested game)." :coerce :long}
   :python {:desc "Python with textworld installed (spec tiers)." :default "python3"}
   :tw_make {:desc "TextWorld's tw-make (compiled tiers)." :default "tw-make"}})

(defn generate-games [opts]
  (fs/create-dirs (:output_dir opts))
  (let [max-attempts (or (:max_attempts opts) (* attempts-per-game (:num_games opts)))
        tools {:python (:python opts) :tw-make (:tw_make opts)}
        workers (max 1 (:workers opts))]
    (loop [specs (game-specs (parse-mix (:tiers opts)) (:seed opts) (:output_dir opts)) built 0 attempts 0]
      (if (or (>= built (:num_games opts)) (>= attempts max-attempts))
        (println (str built " games in " (:output_dir opts) " from " attempts " game indices"))
        (let [batch (take (min workers (- (:num_games opts) built) (- max-attempts attempts)) specs)
              results (mapv (fn [s] (future [s (build-game! s tools)])) batch)
              results (mapv deref results)]
          (doseq [[s err] results]
            (println (if err (str (fs/file-name (:path s)) " skipped: " err) (str (fs/file-name (:path s))))))
          (recur (drop (count batch) specs) (+ built (count (remove second results))) (+ attempts (count batch))))))))

;; --- the dataset -------------------------------------------------------------------------------------

(def splits ["train" "validation" "test"])
(def game-name #"game_(\d+)_([a-z]+)_w(\d+)_o(\d+)_q(\d+)_s(\d+)")
(def episodes-per-compiled-game 2)
(def default-prompt "memory_first")
(def prompts
  (array-map
   "strict_action" [(str "You are an agent in a text adventure game.\n"
                         "Think briefly, then ALWAYS end with exactly one action line in this format:\n"
                         "[ACTION: <command>]\n"
                         "Use short game commands (look, inventory, examine, take, open, close, go north, etc.).")
                    "Start the game and choose the next best action."]
   "memory_first" [(str "You play TextWorld efficiently.\n"
                        "Use memory summaries when useful and avoid repeating failed actions.\n"
                        "Return one final action command in the format [ACTION: <command>].")
                   "What is your next action?"]
   "compact" ["Solve the game step by step.\nOutput must end with [ACTION: <command>]." "Play optimally."]
   "parseable" [(str "You are playing a text-based game.\n"
                     "Final line must be parseable as [ACTION: ...] with only a single command.\n"
                     "Do not output multiple actions.")
                "Continue the game."]))

(defn find-games
  "Games in directories, the compiled .z8 preferred when both forms exist."
  [directories]
  (vals (reduce (fn [games path]
                  (let [stem (str (fs/strip-ext (fs/file-name path)))
                        ext (fs/extension path)
                        m (re-matches game-name stem)]
                    (if (or (nil? m) (not (#{"z8" "json"} ext)) (and (games stem) (:compiled (games stem))))
                      games
                      (let [[_ index tier w o q seed] m]
                        (assoc games stem {:path (str (fs/absolutize path)) :tier tier :index (parse-long index)
                                           :world-size (parse-long w) :nb-objects (parse-long o)
                                           :quest-length (parse-long q) :seed (parse-long seed)
                                           :compiled (= ext "z8") :stem stem})))))
                (array-map)
                (sort (mapcat #(fs/glob % "game_*") directories)))))

(defn split-games
  "Each tier's games into test, validation and train (compiled tiers first,
  then alphabetical, one shuffle stream; a tier keeps a third for training)."
  [games held-out seed]
  (let [by-tier (group-by :tier games)
        rng (pr/random-state seed)
        order (sort-by (fn [t] [(not (:compiled (first (by-tier t)))) t]) (keys by-tier))]
    (reduce (fn [acc tier]
              (let [tier-games (pr/shuffle rng (sort-by #(str (fs/file-name (:path %))) (by-tier tier)))
                    n (min held-out (quot (count tier-games) 3))]
                (-> acc
                    (update "test" into (subvec tier-games 0 n))
                    (update "validation" into (subvec tier-games n (* 2 n)))
                    (update "train" into (subvec tier-games (* 2 n))))))
            {"train" [] "validation" [] "test" []}
            order)))

(defn prompt-variant
  "The prompt variant of a compiled game's episode, drawn from the game seed."
  [game episode]
  (let [rng (pr/random-state (+ (bit-shift-left (:seed game) 8) episode 17))]
    (nth (vec (keys prompts)) (pr/randrange rng (count prompts)))))

(defn make-rows [games]
  (reduce (fn [rows game]
            (reduce (fn [rows episode]
                      (let [variant (if (:compiled game) (prompt-variant game episode) default-prompt)
                            [system user] (prompts variant)]
                        (conj rows {"prompt" [{"role" "system" "content" system} {"role" "user" "content" user}]
                                    "env_class" (if (:compiled game) "textworld" "fast_textworld")
                                    "game_file" (:path game)
                                    "data_source" (:tier game)
                                    "extra_info" (array-map "row_index" (count rows) "game_id" (:stem game)
                                                            "tier" (:tier game) "world_size" (:world-size game)
                                                            "nb_objects" (:nb-objects game)
                                                            "quest_length" (:quest-length game)
                                                            "game_seed" (:seed game) "episode_index" episode
                                                            "prompt_variant" variant)})))
                    rows (range (if (:compiled game) episodes-per-compiled-game 1))))
          [] (sort-by (juxt :tier :index) games)))

(def dataset-spec
  {:games_dirs {:desc "Directories of generated games." :required true :coerce []}
   :output_dir {:desc "Where to write the parquet files." :required true}
   :held_out {:desc "Games per tier in validation and in test." :default 5 :coerce :long}
   :seed {:desc "Split seed." :default 42 :coerce :long}})

(defn build-dataset [opts]
  (let [games (find-games (:games_dirs opts))
        out (:output_dir opts)]
    (fs/create-dirs out)
    (let [stats (into (array-map)
                      (for [[split members] (split-games games (:held_out opts) (:seed opts))]
                        (let [rows (make-rows members)]
                          (pq/write-rows (str (fs/path out (str split ".parquet"))) rows
                                         ["prompt" "env_class" "game_file" "data_source" "extra_info"])
                          (let [s {"games" (into (sorted-map) (frequencies (map :tier members)))
                                   "rows" (into (sorted-map) (frequencies (map #(get % "data_source") rows)))}]
                            (println (str split ": " (count members) " games, " (count rows) " rows " (get s "rows")))
                            [split s]))))]
      (util/write-json (fs/path out "dataset_stats.json") stats))))

(defn -main [command & args]
  (cli/exit-on-error
   #(case command
      "games" (generate-games (cli/parse "usage: jolt rl-games --output_dir DIR --tiers medium:0.3,hard:0.7 --num_games N" games-spec args))
      "dataset" (build-dataset (cli/parse "usage: jolt rl-dataset --games_dirs DIR... --output_dir DIR" dataset-spec args))
      (throw (ex-info (str "unknown command " command " (games or dataset)") {}))))
  (System/exit 0))
