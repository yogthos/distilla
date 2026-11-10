(ns distilla.metrics-test
  "The scorers, on hand-made fixtures and against nltk / rouge_score golden
  values (test/fixtures/metrics.json, from dev/make_fixtures.py)."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [distilla.benchmarks :as b]
            [distilla.dataset :as ds]
            [distilla.helpers :as h]
            [distilla.metrics :as m]
            [distilla.porter :as porter]
            [distilla.util :as util]))

(def golden (delay (util/read-json "test/fixtures/metrics.json")))

(defn- close? [a b] (< (Math/abs (- (double a) (double b))) 1e-9))

(deftest porter-stems-match-nltk
  (doseq [[word stem] (get @golden "stems")]
    (is (= stem (porter/stem word)) word)))

(deftest generation-metrics-match-python
  (let [{:strs [pairs bleu corpus_bleu rouge meteor]} @golden]
    (doseq [[[p r] b1 rg mt] (map vector pairs bleu rouge meteor)]
      (testing (pr-str [p r])
        (let [scores (m/bleu [p] [r])]
          (is (close? (get b1 "bleu1") (scores "bleu1")))
          (is (close? (get b1 "bleu4") (scores "bleu4"))))
        (let [scores (m/rouge [p] [r])]
          (is (close? (get rg "rouge1") (scores "rouge1")))
          (is (close? (get rg "rougeL") (scores "rougeL"))))
        (is (close? mt (m/meteor [p] [r])))))
    (let [scores (m/bleu (map first pairs) (map second pairs))]
      (is (close? (get corpus_bleu "bleu1") (scores "bleu1")))
      (is (close? (get corpus_bleu "bleu4") (scores "bleu4"))))))

(deftest clean-generation
  (is (= "Paris" (m/clean-generation "  Paris<|im_end|>\n")))
  (is (= "a b" (m/clean-generation "<|im_start|>a<|endoftext|> b<|im_sep|>"))))

(deftest normalize-answer
  (doseq [[text expected] [["The Eiffel Tower!" "eiffel tower"]
                           ["  a  Tale of  TWO cities. " "tale of two cities"]
                           ["Anthem" "anthem"]
                           ["" ""]]]
    (is (= expected (m/normalize-answer text)))))

(deftest exact-match
  (is (close? (/ 2.0 3) (m/exact-match ["The Pacific Ocean." "Mark Twain wrote it" "1969"]
                                       ["pacific ocean" "Mark Twain" "1969"]))))

(deftest first-polarity
  (doseq [[text expected] [["Yes. The study shows" "yes"] ["no" "no"] ["Maybe, it depends" "maybe"]
                           ["- Yes." "yes"] ["\"No,\" the authors say" "no"] ["1. Maybe" "maybe"]
                           ["* - No" "no"] ["Perhaps yes" nil] ["" nil] ["   " nil] ["-" nil]]]
    (is (= expected (m/first-polarity text)) (pr-str text))))

(deftest polarity-accuracy-skips-references-without-polarity
  (is (close? (/ 1.0 3) (m/polarity-accuracy ["Yes, clearly." "No." "Maybe" "yes"]
                                             ["yes. Because" "Maybe. Unclear" "unknown" "No"])))
  (is (= 0.0 (m/polarity-accuracy ["yes"] ["unclear"]))))

(deftest token-f1
  (is (close? (/ (* 2 (/ 2.0 3) 1) (+ (/ 2.0 3) 1)) (m/token-f1 ["the cat sat"] ["The cat"])))
  (is (close? (/ (* 2 (/ 1.0 3) 0.5) (+ (/ 1.0 3) 0.5)) (m/token-f1 ["a a b"] ["a c"])))
  (is (= 0.0 (m/token-f1 ["" "x"] ["y" "z"]))))

(deftest bleu-and-rouge
  (let [texts ["the quick brown fox jumps over the lazy dog"]]
    (is (close? 1.0 ((m/bleu texts texts) "bleu1")))
    (is (close? 1.0 ((m/bleu texts texts) "bleu4")))
    (is (= {"rouge1" 1.0 "rougeL" 1.0} (m/rouge texts texts)))
    (is (= {"rouge1" 0.0 "rougeL" 0.0} (m/rouge ["alpha beta"] ["gamma delta"])))
    (is (close? (/ (* 2 1.0 (/ 2.0 3)) (+ 1.0 (/ 2.0 3))) ((m/rouge ["running dogs"] ["the dog runs"]) "rouge1")))
    (is (< ((m/bleu ["the cat"] ["the cat sat on the mat"]) "bleu1") 1.0))))

(deftest generation-metrics-on-identical-texts
  (let [texts ["the quick brown fox jumps over the lazy dog"]
        scores (m/compute-metrics b/generation b/generation-metrics texts texts)]
    (is (= b/generation-metrics (vec (keys scores))))
    (is (> (scores "meteor") 0.99))
    (is (close? 1.0 (scores "f1")))))

(deftest choice-predictions-never-pick-empty-choices
  (let [preds {:rows 2 :cols 2 :data (float-array [1 0 0 1])}
        choice-embeddings [{:rows 2 :cols 2 :data (float-array [0 1 0.9 0.1])} {:rows 1 :cols 2 :data (float-array [1 1])}]]
    (is (= [2 1] (m/choice-predictions preds choice-embeddings [["b" "" "c"] ["" "z"]])))))

(deftest choice-index-ignores-whitespace
  (is (= 2 (m/choice-index ["paris " "" "rome"] "rome")))
  (is (= 0 (m/choice-index ["paris" "rome"] " paris")))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"berlin" (m/choice-index ["paris" "rome"] "berlin"))))

(deftest choice-accuracy-with-hard-subset
  (let [table {"paris" [1.0 0.0] "rome" [0.0 1.0] "berlin" [-1.0 0.0]
               "it is paris" [0.9 0.1] "rome i think" [0.2 0.8] "no idea" [-0.8 -0.1]}
        enc (h/fake-encoder 2 table)
        scores (m/choice-accuracy ["it is paris" " rome i think " "no idea"] ["paris" "berlin" "rome"]
                                  [["paris" "rome" ""] ["paris" "rome" "berlin"] ["rome" "" "berlin"]]
                                  enc [false true true])]
    (is (close? (/ 1.0 3) (scores "accuracy")))
    (is (= 0.0 (scores "accuracy_hard")))
    (is (not (some #{""} (last @(:calls enc)))))))

(deftest compute-metrics-dispatch-and-validation
  (let [scores (m/compute-metrics b/exact-match [b/exact-match "bleu1" "rougeL"] ["Paris"] ["paris"])]
    (is (= [b/exact-match "bleu1" "rougeL"] (vec (keys scores))))
    (is (= 1.0 (scores b/exact-match))))
  (is (= {"accuracy" 1.0} (m/compute-metrics b/polarity [b/accuracy] ["no"] ["No."])))
  (is (thrown? clojure.lang.ExceptionInfo (m/compute-metrics b/polarity [b/exact-match] ["no"] ["no"])))
  (is (thrown? clojure.lang.ExceptionInfo (m/compute-metrics "unknown" [b/accuracy] ["no"] ["no"])))
  (is (thrown? clojure.lang.ExceptionInfo (m/compute-metrics b/choice [b/accuracy] ["no"] ["no"]))))

(deftest write-predictions-and-metrics
  (h/with-temp-dir [dir]
    (let [records [{"question" "q" "answer" "a" "prediction" "a"} {"question" "r" "answer" "b" "prediction" "c"}]]
      (m/write-predictions dir records)
      (let [summary (m/write-metrics dir {"accuracy" (/ 2.0 3)} (count records))]
        (is (= {"accuracy" 66.67 "num_examples" 2} summary))
        (is (= summary (util/read-json (fs/path dir m/metrics-file))))
        (is (= records (util/read-jsonl (fs/path dir m/predictions-file))))))))

(deftest predictions-survive-a-scoring-failure
  (h/with-temp-dir [dir]
    (let [dataset (ds/from-rows (h/make-rows))
          jobs [[(b/benchmarks "pubmedqa") dataset (vec (repeat 3 "yes")) (str (fs/path dir "first"))]
                [(b/benchmarks "lamp7") dataset (vec (repeat 3 "paris")) (str (fs/path dir "second"))]]]
      (binding [m/*score-dataset* (fn [& _] (throw (ex-info "missing scorer data" {})))]
        (is (thrown? clojure.lang.ExceptionInfo (m/score-predictions jobs (constantly nil)))))
      (doseq [[_ _ predictions out] jobs]
        (is (= predictions (mapv #(get % "prediction") (util/read-jsonl (fs/path out m/predictions-file)))))
        (is (not (fs/exists? (fs/path out m/metrics-file))))))))

(deftest score-predictions-end-to-end
  (h/with-temp-dir [dir]
    (let [dataset (ds/from-rows (h/make-rows))
          enc (h/fake-encoder)
          out (str (fs/path dir "pv4"))
          summaries (m/score-predictions [[(b/benchmarks "pv4") dataset ["paris" "shakespeare" "paris"] out]]
                                         (constantly enc))]
      (is (= {"accuracy" 100.0 "accuracy_hard" 100.0 "num_examples" 3} (summaries "pv4")))
      (is (= ["question" "answer" "prediction" "choices" "hard"]
             (vec (keys (first (util/read-jsonl (fs/path out m/predictions-file))))))))))

(deftest check-scoring-inputs
  (let [rows (h/make-rows)]
    (m/check-scoring-inputs (b/benchmarks "pv4") (ds/from-rows rows))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"choices"
                          (m/check-scoring-inputs (b/benchmarks "pv4") (ds/remove-columns (ds/from-rows rows) "choices"))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"marlowe"
                          (m/check-scoring-inputs (b/benchmarks "pv4")
                                                  (ds/from-rows (assoc-in rows [1 "answer"] "marlowe")))))
    (m/check-scoring-inputs (b/benchmarks "lamp7") (ds/from-rows rows))))
