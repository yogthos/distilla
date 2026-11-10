(ns distilla.rl.env
  "TextWorld environments whose completed turns feed a memory
  (SkyRL-Memorilla skyrl_gym/envs/textworld/{base,env,fast_env}.py).

  Two engines play a game:
  - fast_textworld: the in-process simulator (distilla.rl.sim) on a .json
    game spec. A .z8 game file names its .json spec next to it.
  - textworld: TextWorld's Inform7 interpreter on a compiled .z8 game,
    through tools/textworld_bridge.py (needs `pip install textworld`).

  After every turn the environment updates its memory documents; the reward
  of a turn is the game score change TextWorld reports."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [distilla.rl.memory :as memory]
            [jolt.process :as p]))

(def default-max-turns 50)
(def default-action "look")

(def command-hint
  (str "\n\nValid TextWorld commands (use exactly these forms — variations are rejected by the parser):\n"
       "- `look` (describe room — NOT `look around` or `look at <room>`)\n"
       "- `inventory` (list held items)\n"
       "- `examine <object>` (NOT `examine <room>` or `inspect`)\n"
       "- `take <object>` (NOT `pick up`, `grab`, `get`)\n"
       "- `drop <object>`\n"
       "- `open <object>` / `close <object>` (for doors and containers)\n"
       "- `unlock <object> with <key>`\n"
       "- `put <object> in <container>` / `put <object> on <surface>`\n"
       "- `go <direction>` or just `<direction>` (north/south/east/west/up/down) — NOT `move`, `walk`, `head`\n"
       "If a command is rejected (`You can't see any such thing.` or similar), DO NOT repeat it — try a different valid form."))

(def env-defaults
  {"textworld" {:memory-window 1 :max-memory-docs 60 :max-observation-chars 200}
   "fast_textworld" {:memory-window 5 :max-memory-docs 4 :max-observation-chars 0}})

;; --- engines --------------------------------------------------------------------------------

(defprotocol Engine
  (engine-reset [engine] "Start an episode: the opening observation.")
  (engine-step [engine command] "{:observation :reward :done :score :won}")
  (engine-close [engine]))

(defn- json-path [game-file]
  (cond (str/ends-with? game-file ".json") game-file
        (str/ends-with? game-file ".z8") (str (subs game-file 0 (- (count game-file) 3)) ".json")
        :else (str game-file ".json")))

(defrecord SimEngine [path state]
  Engine
  (engine-reset [_]
    (let [load-game (requiring-resolve 'distilla.rl.sim/load-game)
          reset (requiring-resolve 'distilla.rl.sim/reset!)
          sim (load-game path)]
      (reset! state sim)
      (first (reset sim))))
  (engine-step [_ command]
    (let [step (requiring-resolve 'distilla.rl.sim/step!)
          [observation reward done info] (step @state command)]
      {:observation observation :reward (double reward) :done (boolean done)
       :score (long (get info "score" 0)) :won (boolean (get info "won" false))}))
  (engine-close [_] (reset! state nil)))

(def bridge-script
  (delay (let [local (io/file "tools/textworld_bridge.py")]
           (if (.exists local)
             (str local)
             (let [tmp (java.io.File/createTempFile "textworld_bridge" ".py")]
               (spit tmp (slurp (io/resource "textworld_bridge.py")))
               (str tmp))))))

(defn- bridge-request [engine msg]
  (let [{:keys [writer reader game-file]} engine]
    (.write @writer (str (json/write-str msg) "\n"))
    (.flush @writer)
    (let [line (.readLine @reader)]
      (when-not line (throw (ex-info "the TextWorld bridge exited" {:game game-file})))
      (let [r (json/read-str line)]
        (when (get r "error") (throw (ex-info (str "TextWorld bridge: " (get r "error")) {})))
        r))))

(defrecord Inform7Engine [game-file python proc writer reader]
  Engine
  (engine-reset [this]
    (when-not @proc
      (let [pr (p/process {:in :pipe :out :pipe :err :inherit} python "-u" @bridge-script)]
        (reset! proc pr)
        (reset! writer (io/writer (:in pr)))
        (reset! reader (io/reader (:out pr)))))
    (get (bridge-request this {"op" "reset" "game" game-file}) "feedback"))
  (engine-step [this command]
    (let [r (bridge-request this {"op" "step" "command" command})]
      {:observation (get r "feedback") :reward (double (get r "reward" 0.0)) :done (boolean (get r "done"))
       :score (long (get r "score" 0)) :won (boolean (get r "won"))}))
  (engine-close [_]
    (when-let [w @writer]
      (try (.write w "{\"op\": \"close\"}\n") (.flush w) (.close w) (catch Exception _ nil))
      (reset! writer nil)
      (reset! proc nil))))

(defn engine
  "The engine for an env class and game file. opts :inform7 (\"bridge\",
  default, or \"simulator\" to play .z8 games through their .json spec)
  and :python (the interpreter that has textworld)."
  [env-class game-file {:keys [inform7 python] :or {inform7 "bridge" python "python3"}}]
  (if (and (= env-class "textworld") (= inform7 "bridge"))
    (->Inform7Engine game-file python (atom nil) (atom nil) (atom nil))
    (->SimEngine (json-path game-file) (atom nil))))

;; --- the environment --------------------------------------------------------------------------

(defn make-env
  "An environment for one dataset row.
  opts: :env-class (\"textworld\" or \"fast_textworld\"), :game-file,
  :max-turns (50), :enable-memory (true), and per class overrides of
  :memory-window :max-memory-docs :max-observation-chars, plus engine opts."
  [{:keys [env-class game-file max-turns enable-memory] :or {max-turns default-max-turns enable-memory true} :as opts}]
  (when-not game-file (throw (ex-info "an environment needs a game file" {})))
  (let [settings (merge (env-defaults env-class (env-defaults "textworld"))
                        (select-keys opts [:memory-window :max-memory-docs :max-observation-chars]))]
    {:env-class env-class
     :game-file game-file
     :max-turns max-turns
     :enable-memory enable-memory
     :settings settings
     :engine (engine env-class game-file opts)
     :state (atom nil)}))

(defn- new-memory [{:keys [settings]}]
  (memory/trajectory-memory (:memory-window settings) (:max-memory-docs settings) (:max-observation-chars settings)))

(defn init!
  "Reset the game: the prompt messages (system turn extended for the
  simulator) followed by the opening observation as a user message."
  [{:keys [env-class engine state] :as env} prompt]
  (let [observation (str/trim (engine-reset engine))
        messages (vec prompt)
        messages (if (and (seq messages) (= env-class "fast_textworld"))
                   (update-in messages [0 :content] #(str (str/trimr %) command-hint))
                   messages)]
    (reset! state {:observation observation :turns 0 :score 0 :won false :memory (new-memory env)})
    (conj messages {:role "user" :content observation})))

(defn documents
  "The current memory documents (none when memory is off)."
  [{:keys [state enable-memory]}]
  (if enable-memory (vec (remove str/blank? (memory/documents (:memory @state)))) []))

(defn step!
  "Play one turn from a raw model response: {:observations :reward :done :metadata}."
  [{:keys [engine state max-turns]} action]
  (let [{:keys [observation turns memory]} @state
        turns (inc turns)
        command (let [c (memory/parse-action action)] (if (str/blank? c) default-action c))
        {obs :observation :keys [reward done score won]} (engine-step engine command)]
    (memory/add-turn! memory {:turn turns :observation observation :action command :reward reward :score score})
    (when (or done (>= turns max-turns)) (memory/flush! memory))
    (let [done (or done (>= turns max-turns))]
      (swap! state assoc :observation (str/trim obs) :turns turns :score score :won won)
      {:observations (if done [] [{:role "user" :content (str/trim obs)}])
       :reward reward
       :done done
       :metadata {:action command :score score :won won}})))

(defn metrics [{:keys [state]}]
  (let [{:keys [turns won score]} @state] {"steps" turns "won" won "score" score}))

(defn close! [{:keys [engine]}] (engine-close engine))
