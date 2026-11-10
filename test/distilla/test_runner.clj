(ns distilla.test-runner
  (:require [clojure.test :as t]
            distilla.data-test
            distilla.formats-test
            distilla.llama-test
            distilla.memory-test
            distilla.metrics-test
            distilla.model-test
            distilla.registry-test
            distilla.train-test
            distilla.pv4-test
            distilla.rl-test
            distilla.rl-sim-test))

(def namespaces
  '[distilla.formats-test
    distilla.memory-test
    distilla.metrics-test
    distilla.data-test
    distilla.model-test
    distilla.registry-test
    distilla.train-test
    distilla.pv4-test
    distilla.rl-test
    distilla.rl-sim-test
    distilla.llama-test])

(defn -main [& args]
  (let [selected (if (seq args) (map symbol args) namespaces)
        {:keys [fail error]} (apply t/run-tests selected)]
    (System/exit (if (zero? (+ fail error)) 0 1))))
