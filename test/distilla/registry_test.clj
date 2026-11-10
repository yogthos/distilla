(ns distilla.registry-test
  "The benchmark registry, the training recipes, the chat templates and the
  evaluation targets."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [distilla.benchmarks :as b]
            [distilla.evaluate :as evaluate]
            [distilla.baselines :as baselines]
            [distilla.tokenizer :as tok]
            [distilla.util :as util]))

(def expected-names ["pmv2" "pv4" "factkg" "triviaqa" "narrativeqa" "pubmedqa" "lamp4" "lamp7"])

(deftest registry-contents
  (is (= expected-names b/benchmark-names))
  (doseq [[name benchmark] b/benchmarks]
    (is (= name (:name benchmark)))
    (is (= "test" (:split benchmark)))
    (is (seq (:config benchmark)))))

(deftest benchmark-protocol
  (doseq [benchmark b/benchmark-list]
    (testing (:name benchmark)
      (is (some #{(:scoring benchmark)} b/scorings))
      (is (and (seq (:metrics benchmark)) (apply distinct? (:metrics benchmark))))
      (is (every? (set (b/metrics-by-scoring (:scoring benchmark))) (:metrics benchmark)))
      (is (< 0 (:max-new-tokens benchmark) (:max-length benchmark)))
      (is (not (str/blank? (:system-prompt benchmark))))
      (let [baseline (:baseline benchmark)]
        (is (not (str/blank? (:system-prompt baseline))))
        (is (pos? (:max-new-tokens baseline)))
        (is (> (:full-context-length baseline) (:max-new-tokens baseline)))
        (when-let [factor (:yarn-factor baseline)]
          (is (= (:full-context-length baseline) (int (* factor b/yarn-original-max-position-embeddings))))
          (is (> (:full-context-length baseline) b/retrieval-context-length)))))))

(deftest scoring-assignments
  (is (= #{"pmv2" "pv4" "factkg"} (set (for [[n bm] b/benchmarks :when (= b/choice (:scoring bm))] n))))
  (is (some #{"accuracy_hard"} (:metrics (b/benchmarks "pv4"))))
  (is (= b/trivia-suffix (:suffix (b/benchmarks "triviaqa")) (get-in b/benchmarks ["triviaqa" :baseline :suffix])))
  (is (str/ends-with? b/trivia-suffix "Your answer:")))

(deftest lookup
  (is (= expected-names (map :name (b/get-benchmarks "all"))))
  (is (= [(b/benchmarks "lamp7")] (b/get-benchmarks "lamp7")))
  (is (thrown? clojure.lang.ExceptionInfo (b/get-benchmarks "unknown"))))

(deftest validation
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"^x: Unknown scoring"
                        (b/benchmark {:name "x" :config "x" :split "test" :system-prompt "s" :max-length 128
                                      :max-new-tokens 8 :scoring "bogus" :metrics ["accuracy"]})))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"^x: Metrics"
                        (b/benchmark {:name "x" :config "x" :split "test" :system-prompt "s" :max-length 128
                                      :max-new-tokens 8 :scoring b/choice :metrics ["bleu1"]})))
  (b/validate-metrics b/generation b/generation-metrics)
  (is (thrown? clojure.lang.ExceptionInfo (b/validate-metrics b/generation [b/exact-match]))))

;; --- recipes ---------------------------------------------------------------------------

(defn- recipe [name] (util/read-recipe (str "configs/" name ".yaml")))

(deftest recipes-parse
  (doseq [path (fs/glob "configs" "*.yaml")]
    (let [r (util/read-recipe (str path))]
      (is (seq (get r "datasets")) (str path))
      (is (get r "output_dir") (str path)))))

(deftest recipe-chain
  (let [[s1 s2 s3] (map recipe ["stage1_enwiki" "stage2_mixture" "stage3_multitask"])]
    (is (str/starts-with? (get s2 "init_memory") (str (get s1 "output_dir") "/")))
    (is (str/starts-with? (get s3 "init_memory") (str (get s2 "output_dir") "/")))
    (doseq [name ["personalization_pv4" "personalization_pmv2"]]
      (is (str/starts-with? (get (recipe name) "init_memory") (str (get s3 "output_dir") "/")))))
  (let [factkg (recipe "factkg")]
    (is (true? (get factkg "retrieval_init")))
    (is (nil? (get factkg "init_memory")))
    (is (not (str/includes? (get factkg "system_prompt") "\n")))
    (is (str/ends-with? (get factkg "system_prompt") "with no other text."))))

(deftest evaluation-targets
  (testing "a recipe's system prompt and overrides apply to every benchmark"
    (let [targets (evaluate/resolve-targets {:benchmark "factkg" :split "test"
                                             :system_prompt_from "configs/factkg.yaml" :max_new_tokens 4})]
      (is (= (get (recipe "factkg") "system_prompt") (:system-prompt (first targets))))
      (is (= 4 (:max-new-tokens (first targets))))))
  (testing "another split renames the target"
    (is (= "lamp7-validation" (:name (first (evaluate/resolve-targets {:benchmark "lamp7" :split "validation"}))))))
  (testing "a custom config takes the scoring's metrics"
    (let [[t] (evaluate/resolve-targets {:config "my_task" :split "test" :scoring b/polarity})]
      (is (= ["accuracy"] (:metrics t)))
      (is (= 2048 (:max-length t))))))

(deftest baseline-helpers
  (is (= "rag_top5" (baselines/method-label "rag" 5)))
  (is (= [131072 4.0] (baselines/context-window (b/benchmarks "triviaqa") "full_context")))
  (is (= [b/retrieval-context-length nil] (baselines/context-window (b/benchmarks "triviaqa") "rag")))
  (is (= "a\n\nb\n\nq?s" (baselines/user-message ["a" "b"] "q?" "s")))
  (let [summary (baselines/context-summary [10 20 30 40] 64 25)]
    (is (= 2 (summary "num_truncated")))
    (is (= 50.0 (summary "fit_rate")))
    (is (= 20 (summary "tokens_dropped")))
    (is (= 25.0 (summary "p50_prompt_tokens")))))

;; --- chat templates ---------------------------------------------------------------------------

(deftest qwen3-template
  (let [messages [{:role "system" :content "S"} {:role "user" :content "Q"}]]
    (is (= "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
           (tok/qwen3 messages true)))
    (is (= (str "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\nA<|im_end|>\n")
           (tok/qwen3 (conj messages {:role "assistant" :content "A"}) false)))
    (is (= "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n"
           (tok/chatml messages true)))))
