(ns distilla.model-test
  "The memory-conditioned model with a fake decoder engine: prompts are cut at
  the memory placeholders and carry the memory module's output there."
  (:require [clojure.test :refer [deftest is testing]]
            [distilla.data :as data]
            [distilla.helpers :as h]
            [distilla.memory :as memory]
            [distilla.model :as model]
            [distilla.tensor :as t]
            [distilla.tokenizer :as tok]))

(def hidden-size 16)
(def num-memories 2)
(def question "what is the capital of france")
(def documents ["paris is the capital" "france"])

(defrecord FakeEngine [text requests]
  model/Engine
  (generate-text [_ segments opts]
    (swap! requests conj {:segments segments :opts opts})
    text))

(defn- make-memory []
  (let [cfg (memory/config {:embedding-dim h/embedding-dim :output-dim hidden-size :num-memories num-memories
                            :num-heads 2 :dropout 0.0})]
    (memory/from-state cfg (h/random-state cfg 21))))

(defn- make-model [enc]
  (model/memory-model {:memory (make-memory) :tokenizer (h/word-tokenizer)
                       :engine (->FakeEngine " paris \n" (atom [])) :encoder enc}))

(deftest prompt-segments
  (let [mem (t/matrix 3 2 (float-array [1 2 3 4 5 6]))
        segments (model/prompt-segments [5 6 -1 -1 -1 7 8] -1 mem)]
    (is (= [:tokens [5 6]] (first segments)))
    (is (= [1.0 2.0 3.0 4.0 5.0 6.0] (map double (:data (second (second segments))))))
    (is (= [:tokens [7 8]] (nth segments 2))))
  (is (thrown? clojure.lang.ExceptionInfo (model/prompt-segments [1 -1 2] -1 (t/matrix 2 2)))))

(deftest answer-fills-the-memory-positions
  (let [enc (h/fake-encoder)]
    (with-open [m (make-model enc)]
      (is (= "paris" (model/answer m question documents {:max-new-tokens 8})))
      (is (= (conj documents question) (last @(:calls enc))))
      (let [tokenizer (:tokenizer m)
            prompt (tok/apply-chat-template tokenizer (data/build-messages question data/default-system-prompt num-memories) true)
            ids (tok/encode tokenizer prompt)
            vectors (mapv #(h/vector-of enc %) (conj documents question))
            expected (memory/forward (:memory m) (t/from-rows (pop vectors)) (peek vectors))
            [{:keys [segments opts]}] @(:requests (:engine m))
            mem-id (tok/memory-token-id tokenizer)
            [before-mem] (split-with #(not= mem-id %) ids)]
        (is (= 3 (count segments)))
        (is (= [:tokens (vec before-mem)] (first segments)))
        (is (= :embeddings (first (second segments))))
        (is (h/approx= (vec (:data (second (second segments)))) (vec (:data expected)) 1e-6))
        (is (= (vec (drop (+ (count before-mem) num-memories) ids)) (second (nth segments 2))))
        (is (= 8 (:max-new-tokens opts)))
        (is (= 0.0 (:temperature opts)))))))

(deftest answer-needs-an-encoder
  (with-open [m (make-model nil)]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"encoder" (model/answer m question documents)))))

(deftest batched-generation-strips-padding
  (with-open [m (make-model nil)]
    (let [tokenizer (:tokenizer m)
          mem-id (tok/memory-token-id tokenizer)
          pad (tok/pad-id tokenizer)
          batch {:input-ids [[pad pad 5 mem-id mem-id 6] [4 5 mem-id mem-id 6 7]]
                 :attention-mask [[0 0 1 1 1 1] [1 1 1 1 1 1]]
                 :doc-embeds [(t/matrix 3 h/embedding-dim (h/normals 1 24)) (t/matrix 3 h/embedding-dim (h/normals 2 24))]
                 :doc-padding-mask [[false false true] [false false false]]
                 :question-embeds [(h/normals 3 h/embedding-dim) (h/normals 4 h/embedding-dim)]}]
      (is (= [" paris \n" " paris \n"] (model/generate m batch)))
      (let [[first-request second-request] @(:requests (:engine m))]
        (is (= [:tokens [5]] (first (:segments first-request))))
        (is (= [:tokens [4 5]] (first (:segments second-request))))))))

(deftest resolve-checkpoint-keeps-local-folders
  (h/with-temp-dir [dir]
    (with-open [m (make-memory)]
      (memory/save m (str dir "/pmv2")))
    (with-redefs [distilla.hub/snapshot-download (fn [& _] (throw (AssertionError. "unexpected download")))]
      (is (= (str dir "/pmv2") (model/resolve-checkpoint dir "pmv2" nil)))
      (with-open [m (memory/from-pretrained (model/resolve-checkpoint dir "pmv2" nil))]
        (is (= num-memories (memory/num-memories m)))))))
