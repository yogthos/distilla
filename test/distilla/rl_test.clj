(ns distilla.rl-test
  "The RL port: game parameters, dataset rows, aggregation, action parsing
  and memory documents against memorilla's and SkyRL-Memorilla's Python
  (test/fixtures/rl/python.json from dev/make_rl_fixtures.py); rollouts, GRPO
  advantages and the policy-gradient step with a scripted game and fakes."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [distilla.helpers :as h]
            [distilla.memory :as memory]
            [distilla.optim :as optim]
            [distilla.rl.aggregate :as aggregate]
            [distilla.rl.dataset :as dataset]
            [distilla.rl.env :as env]
            [distilla.rl.grpo :as grpo]
            [distilla.rl.memory :as rlmem]
            [distilla.tensor :as t]
            [distilla.tokenizer :as tok]
            [distilla.util :as util]))

(def python (delay (util/read-json "test/fixtures/rl/python.json")))

(deftest game-parameters-match-python
  (let [mix (dataset/parse-mix "medium:0.3,hard:0.4,long:0.2,extreme:0.1")
        specs (take 40 (dataset/game-specs mix 42 "games"))]
    (doseq [[ours theirs] (map vector specs (get @python "specs"))]
      (is (= (get theirs "name") (str (fs/file-name (:path ours)))))
      (is (= [(theirs "tier") (theirs "world_size") (theirs "nb_objects") (theirs "quest_length")
              (theirs "quest_breadth") (theirs "seed")]
             [(:tier ours) (:world-size ours) (:nb-objects ours) (:quest-length ours) (:quest-breadth ours) (:seed ours)])))))

(deftest dataset-rows-match-python
  (h/with-temp-dir [dir]
    (doseq [s (get @python "specs")]
      (spit (str dir "/" (s "name")) "")
      (when (str/ends-with? (s "name") ".z8")
        (spit (str dir "/" (str/replace (s "name") ".z8" ".json")) "")))
    (let [splits (dataset/split-games (dataset/find-games [dir]) 3 42)]
      (doseq [split dataset/splits]
        (is (= (get-in @python ["dataset" split])
               (mapv #(update % "game_file" (fn [f] (str (fs/file-name f)))) (dataset/make-rows (splits split))))
            split)))))

(defn- close-maps? [a b]
  (and (= (set (keys a)) (set (keys b)))
       (every? (fn [k] (let [x (get a k) y (get b k)]
                         (if (map? x) (close-maps? x y) (< (Math/abs (- (double x) (double y))) 1e-9))))
               (keys a))))

(deftest aggregation-matches-python
  (let [rollouts (get @python "rollouts")]
    (is (close-maps? (get @python "scores") (aggregate/score-rollouts rollouts)))
    (is (close-maps? (get @python "average")
                     (aggregate/average [(aggregate/score-rollouts rollouts) (aggregate/score-rollouts (subvec rollouts 0 12))])))
    (is (= "all" (last (keys (aggregate/average [(aggregate/score-rollouts rollouts)])))))))

(deftest actions-and-documents-match-python
  (doseq [[response expected] (get @python "actions")]
    (is (= expected (rlmem/parse-action response)) (pr-str response)))
  (let [mem (rlmem/trajectory-memory 2 3 15)]
    (doseq [t (range 1 8)]
      (rlmem/add-turn! mem {:turn t :observation (str "Room " t ": a long observation text") :action (str "go " t)
                            :reward (if (zero? (mod t 3)) 1.0 0.0) :score (quot t 3)}))
    (rlmem/flush! mem)
    (is (= (get @python "documents") (rlmem/documents mem)))))

;; --- rollouts with a scripted game ---------------------------------------------------------------

(defrecord ScriptedEngine [script position]
  env/Engine
  (engine-reset [_] (reset! position 0) "You are in a kitchen. There is a key here.")
  (engine-step [_ command]
    (let [i (swap! position inc)
          won (= command "take key")]
      {:observation (if won "You take the key. You win!" (str "Nothing happens (" i ")."))
       :reward (if won 1.0 0.0) :done won :score (if won 1 0) :won won}))
  (engine-close [_] nil))

(defn- scripted-generate
  "A fake agent: replies with a scripted action, ends with the stop token."
  [replies]
  (let [i (atom -1)]
    (fn [segments _opts]
      (let [text (nth replies (min (swap! i inc) (dec (count replies))))]
        {:text text :tokens [7 8] :stop-token 2 :stop-reason "stop" :segments segments}))))

(def row {"prompt" [{"role" "system" "content" "You play TextWorld."} {"role" "user" "content" "What is your next action?"}]
          "env_class" "textworld" "game_file" "kitchen.z8" "data_source" "medium"
          "extra_info" {"row_index" 0}})

(defn- policy [replies module]
  (let [tokenizer (h/word-tokenizer)]
    {:generate (scripted-generate replies) :tokenizer tokenizer :eos (get-in tokenizer [:vocab "<|im_end|>"])
     :memory-module module :encoder (h/fake-encoder) :num-memories 2 :rng (memory/rng 1)}))

(defn- tiny-module []
  (grpo/init-memory-module {:embedding-dim 8 :output-dim 4 :num-memories 2 :num-heads 2 :dropout 0.0} 3))

(deftest rollout-follows-the-token-in-token-out-layout
  (with-redefs [env/engine (fn [& _] (->ScriptedEngine nil (atom 0)))]
    (with-open [m (tiny-module)]
      (let [seen (atom [])
            p (policy ["look around [ACTION: look]" "[ACTION: take key]"] m)
            p (update p :generate (fn [g] (fn [segs o] (swap! seen conj segs) (g segs o))))
            traj (grpo/rollout p row {:max-turns 5})]
        (is (= [0.0 1.0] (:rewards traj)))
        (is (= 1.0 (:return traj)))
        (is (= "stop" (:stop-reason traj)))
        (is (= (count (:response-ids traj)) (count (:loss-mask traj))))
        (testing "each reply is its tokens plus the stop token, masked in; observations masked out"
          (is (= [7 8 2] (subvec (:response-ids traj) 0 3)))
          (is (= [1 1 1] (subvec (:loss-mask traj) 0 3)))
          (is (zero? (nth (:loss-mask traj) 3))))
        (testing "no memory before the first document, then memory vectors before the prompt"
          (is (= [:tokens] (mapv first (first @seen))))
          (is (= [:embeddings :tokens] (mapv first (second @seen))))
          (is (= 2 (:rows (second (first (second @seen)))))))
        (is (= 2 (count (:documents traj))))
        (is (= 1.0 (reduce + (:token-rewards traj))))))))

(deftest rollout-stops-at-the-turn-limit-and-input-budget
  (with-redefs [env/engine (fn [& _] (->ScriptedEngine nil (atom 0)))]
    (let [traj (grpo/rollout (policy ["[ACTION: dance]"] nil) row {:max-turns 3})]
      (is (= 3 (count (:rewards traj))))
      (is (= 0.0 (:return traj)))
      (is (empty? (:documents traj))))
    (let [traj (grpo/rollout (policy ["[ACTION: dance]"] nil) row {:max-turns 50 :max-input-length 40})]
      (is (= "length" (:stop-reason traj))))))

;; --- GRPO -------------------------------------------------------------------------------------------

(deftest group-normalised-advantages
  (let [a (grpo/advantages [1.0 0.0 0.0 2.0 2.0 2.0] 3)]
    (is (< (Math/abs (- (nth a 0) (/ (- 1.0 (/ 1.0 3)) (+ (Math/sqrt (/ (+ 4/9 1/9 1/9) 2)) 1e-6)))) 1e-9))
    (is (every? #(< (Math/abs %) 1e-9) (subvec a 3)) "equal returns carry no signal")
    (is (< (Math/abs (- 3.0 (first (grpo/advantages [3.0] 1)))) 1e-5) "a group of one: mean 0, std 1, as SkyRL")))

(deftest training-sequences-score-the-replies
  (let [traj {:prompt-ids [10 11] :response-ids [7 8 2 20 21] :loss-mask [1 1 1 0 0] :documents ["d"]}
        {:keys [tokens mem-index targets n-response]} (grpo/training-sequence traj 2)]
    (is (= [0 0 10 11 7 8 2 20 21] tokens))
    (is (= [0 1 -1 -1 -1 -1 -1 -1 -1] mem-index))
    (is (= [-1 -1 -1 7 8 2 -1 -1 -1] targets))
    (is (= 3 n-response))))

(deftest policy-gradient-step-updates-the-memory
  (with-redefs [env/engine (fn [& _] (->ScriptedEngine nil (atom 0)))]
    (with-open [m (tiny-module)]
      (let [replies (cycle ["[ACTION: take key]" "[ACTION: look]" "[ACTION: look]" "[ACTION: look]"])
            calls (atom [])
            p (assoc (policy ["[ACTION: look]" "[ACTION: take key]"] m) :optimizer (optim/adamw m {}))
            p (assoc p :generate (let [i (atom -1)]
                                   (fn [_ _] {:text (nth replies (swap! i inc)) :tokens [7] :stop-token 2 :stop-reason "stop"})))
            decoder (fn [{:keys [memory targets]} grad?]
                      (swap! calls conj (count (filter #(>= % 0) targets)))
                      {:loss 1.0 :n-targets (count (filter #(>= % 0) targets))
                       :grad (t/matrix (:rows memory) (:cols memory) (float-array (repeat (* (:rows memory) (:cols memory)) 0.1)))})
            before (vec (memory/param-floats m "memory_projection.bias"))
            result (grpo/train-step! p decoder [row] {:n-samples 4 :micro-batch 2 :lr 1e-2 :sampling {:max-turns 3}})]
        (is (= 1 (:step result)))
        (is (= 4 (count (:trajectories result))))
        (is (pos? (count @calls)) "trajectories with a non-zero advantage are scored")
        (is (not= before (vec (memory/param-floats m "memory_projection.bias"))))))))

(deftest evaluation-dumps-aggregate
  (h/with-temp-dir [dir]
    (with-redefs [env/engine (fn [& _] (->ScriptedEngine nil (atom 0)))]
      (grpo/evaluate! (policy ["[ACTION: take key]"] nil) [row (assoc row "data_source" "hard" "extra_info" {"row_index" 1})]
                      {:n-samples 3 :sampling {:max-turns 3} :dump-dir dir})
      (let [metrics (aggregate/average [(aggregate/score-rollouts (aggregate/load-rollouts dir))])]
        (is (= #{"medium" "hard" "all"} (set (keys metrics))))
        (is (= 1.0 (get-in metrics ["all" "avg@k"])))
        (is (= 6.0 (get-in metrics ["all" "rollouts"])))))))
