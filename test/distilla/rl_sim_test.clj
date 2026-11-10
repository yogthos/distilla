(ns distilla.rl-sim-test
  "Replays the transcripts that dev/make_rl_sim_fixtures.py recorded from the Python
  FastTextWorldSimulator and checks the port answers every step identically."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [distilla.rl.sim :as sim]))

(def ^:private fixture-dir "test/fixtures/rl")

(defn- transcripts []
  (->> (.listFiles (io/file fixture-dir))
       (map #(.getName %))
       (filter #(str/ends-with? % ".transcript.json"))
       sort))

(defn- numbers-as-doubles
  "Python writes ints and floats; compare every number as a double."
  [x]
  (cond (number? x) (double x)
        (map? x) (into {} (map (fn [[k v]] [k (numbers-as-doubles v)])) x)
        (sequential? x) (mapv numbers-as-doubles x)
        :else x))

(defn- same? [expected actual]
  (= (numbers-as-doubles expected) (numbers-as-doubles actual)))

(defn- replay
  "Replays one transcript; returns the number of steps checked."
  [file]
  (let [t (json/read-str (slurp (str fixture-dir "/" file)))
        observations (get t "observations")
        privileged (get t "privileged")
        game (sim/load-game (str fixture-dir "/" (get t "game")))
        checked (volatile! 0)]
    (doseq [[e episode] (map-indexed vector (get t "episodes"))]
      (let [{:strs [reset steps]} episode
            [obs info] (sim/reset! game)
            where (str file " episode " e)]
        (is (= (nth observations (get reset "obs")) obs) (str where " reset observation"))
        (is (same? (get reset "info") info) (str where " reset info"))
        (is (same? (nth privileged (get reset "privileged")) (sim/privileged-state game))
            (str where " reset privileged state"))
        (doseq [[i step] (map-indexed vector steps)]
          (let [[obs reward done info] (sim/step! game (get step "cmd"))
                where (str where " step " i " " (pr-str (get step "cmd")))]
            (is (= (nth observations (get step "obs")) obs) (str where " observation"))
            (is (== (get step "reward") reward) (str where " reward"))
            (is (= (get step "done") done) (str where " done"))
            (is (same? (get step "info") info) (str where " info"))
            (when-let [p (get step "privileged")]
              (is (same? (nth privileged p) (sim/privileged-state game)) (str where " privileged state")))
            (is (= (get-in step ["info" "score"]) (sim/score game)) (str where " score"))
            (is (= (get-in step ["info" "won"]) (sim/won? game)) (str where " won?"))
            (is (= (get step "done") (sim/done? game)) (str where " done?"))
            (is (= (get-in step ["info" "moves"]) (sim/moves game)) (str where " moves")))
          (vswap! checked inc))))
    @checked))

(deftest transcripts-replay-identically
  (let [files (transcripts)]
    (is (<= 4 (count files)) "the fixtures are in test/fixtures/rl")
    (doseq [file files]
      (testing file
        (is (pos? (replay file)))))))

(deftest feedback-is-empty
  ;; the Python property reads an attribute that is never set
  (let [game (sim/load-game (str fixture-dir "/two_quests.json"))]
    (sim/reset! game)
    (sim/step! game "look")
    (is (= "" (sim/feedback game)))))
