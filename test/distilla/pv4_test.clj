(ns distilla.pv4-test
  "PersonalizationV4: generation with a fake API and dataset building, checked
  against the Python pipeline's own outputs on the same inputs
  (test/fixtures/pv4, from dev/make_pv4_fixtures.py), and the seeded random
  generators against CPython and numpy."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [distilla.csv :as csv]
            [distilla.helpers :as h]
            [distilla.openai :as openai]
            [distilla.parquet :as pq]
            [distilla.pv4.build :as build]
            [distilla.pv4.generate :as gen]
            [distilla.pyrandom :as pr]
            [distilla.util :as util]))

(def fixtures "test/fixtures/pv4")

(def question-block
  (str "<QUESTION>\n{stem}\n<CHOICE_A>\nOption alpha for {stem}\n<CHOICE_B>\nOption bravo, with a comma, for {stem}\n"
       "<CHOICE_C>\nOption \"charlie\" for {stem}\n<CHOICE_D>\nOption delta for {stem}\n<CHOICE_E>\nOption echo for {stem}\n"
       "<CORRECT_CHOICE>\n{letter}\n<RATIONALE>\nBecause of the persona.\n<WORD_COUNTS>\nA: 4 | B: 4 | C: 4 | D: 4 | E: 4\n"))

(defn- starts-like [prompt template] (str/starts-with? prompt (subs template 0 40)))

(defn- fake-api
  "Answers each prompt by its template, as the Python fixture's FakeResponses
  does, and records [model effort] per call. fail? decides which prompts fail."
  ([] (fake-api (constantly false)))
  ([fail?]
   (let [calls (atom [])
         t (fn [k] (slurp (clojure.java.io/resource (str "pv4/" k ".txt"))))]
     {:calls calls
      :client (openai/client
               {:max-retries 1
                :complete-fn
                (fn [prompt model effort]
                  (swap! calls conj [model effort])
                  (when (fail? prompt model) (throw (ex-info "unavailable" {})))
                  (cond
                    (starts-like prompt (t "categories")) "1. Teaching Style & Students\n2. Running Routine\n3. Food\n"
                    (starts-like prompt (t "topics")) (str/join "\n" (for [i (range 12)] (str (inc i) ". Topic number " i ".")))
                    (starts-like prompt (t "chat"))
                    (let [topic (first (str/split (second (str/split prompt #"2\. \*\*Scenario:\*\*\n" 2)) #"\n" 2))]
                      (str "  Leo: Help with " topic "\nAssistant: Sure.  \n"))
                    :else
                    (let [category (first (str/split (second (str/split prompt #"following category:\n\*\*" 2)) #"\*\*" 2))]
                      (str (str/join "\n" (for [i (range 5)]
                                            (-> question-block
                                                (str/replace "{stem}" (str category " question " i "?"))
                                                (str/replace "{letter}" (str (nth "BDAEC" (mod i 5)))))))
                           "<QUESTION>\nA question without choices.\n"))))})})))

(def config (merge gen/defaults {:num-categories 3 :num-topics 12 :num-questions 5}))

(deftest generation-matches-python
  (h/with-temp-dir [dir]
    (let [{:keys [client calls]} (fake-api)]
      (doseq [id [3 4 7]]
        (is (gen/run-user! client config (str fixtures "/personas/user_" id ".txt") (str dir "/user_" id))))
      (doseq [id [3 4 7]
              file ["categories.txt" "chat_topics.txt" "qa.csv" "chats/0.txt" "chats/11.txt" "qa/0.txt" "qa/2.txt"]]
        (is (= (slurp (str fixtures "/users/user_" id "/" file)) (slurp (str dir "/user_" id "/" file)))
            (str "user_" id "/" file)))
      (is (= (* 3 12) (count (filter #{[(:chat-model config) (:chat-reasoning-effort config)]} @calls))))
      (is (= (* 3 5) (count (filter #{[(:model config) (:reasoning-effort config)]} @calls)))))))

(deftest generation-is-resumable
  (h/with-temp-dir [dir]
    (let [user (str dir "/user_3")
          persona (str fixtures "/personas/user_3.txt")]
      (gen/run-user! (:client (fake-api)) config persona user)
      (let [table (slurp (str user "/qa.csv"))
            {:keys [client calls]} (fake-api)]
        (is (gen/run-user! client config persona user))
        (is (empty? @calls))
        (fs/delete (str user "/chats/5.txt"))
        (fs/delete (str user "/qa.csv"))
        (is (gen/run-user! client config persona user))
        (is (= [[(:chat-model config) (:chat-reasoning-effort config)]] @calls))
        (is (= table (slurp (str user "/qa.csv"))))))))

(deftest failed-requests-leave-the-user-incomplete
  (h/with-temp-dir [dir]
    (let [user (str dir "/user_3")
          persona (str fixtures "/personas/user_3.txt")]
      (is (not (gen/run-user! (:client (fake-api (fn [_ model] (= model (:chat-model config))))) config persona user)))
      (is (not (fs/exists? (str user "/qa.csv"))))
      (is (empty? (fs/glob (str user "/chats") "*.txt")))))
  (h/with-temp-dir [dir]
    (let [user (str dir "/user_3")
          persona (str fixtures "/personas/user_3.txt")
          categories? (fn [prompt _] (str/includes? prompt "data analysis specialist"))]
      (is (not (gen/run-user! (:client (fake-api categories?)) config persona user)))
      (is (not (fs/exists? (str user "/categories.txt"))))
      (is (gen/run-user! (:client (fake-api)) config persona user))
      (is (fs/exists? (str user "/qa.csv"))))))

(deftest parse-and-shuffle-keep-answers
  (let [text (str (-> question-block (str/replace "{stem}" "Stem?") (str/replace "{letter}" "C"))
                  "<QUESTION>\nA question without choices or an answer.\n")
        records (gen/parse-questions text)]
    (is (= 2 (count records)))
    (is (= "Option \"charlie\" for Stem?" (get (first records) "choice_c")))
    (is (str/starts-with? (get (first records) "rationale") "Because of the persona."))
    (let [[shuffled & more] (gen/shuffle-choices records (pr/random-state 1))]
      (is (empty? more))
      (is (= "Option \"charlie\" for Stem?"
             (get shuffled (str "choice_" (str/lower-case (get shuffled "correct_choice")))))))))

(deftest build-matches-python
  (h/with-temp-dir [dir]
    (let [expected (util/read-json (str fixtures "/expected.json"))
          counts (build/build! {:users-dir (str fixtures "/users") :output-dir dir
                                :train-users (build/parse-user-range "3-4") :test-users (build/parse-user-range "7")
                                :hard-subset (str fixtures "/hard.csv") :id-column "collection_id"})]
      (is (= (get expected "counts") counts))
      (doseq [split build/splits]
        (let [file (str dir "/" split ".parquet")]
          (is (= ["collection_id" "question" "answer" "choices" "documents" "hard"] (pq/column-names file)))
          (is (= (get-in expected ["rows" split]) (pq/read-rows [file])) split)))
      (is (= [["3" "training"] ["4" "training"] ["7" "evaluation"]]
             (mapv (juxt #(get % "user_id") #(get % "role"))
                   (csv/read-maps (slurp (str dir "/" build/user-splits-file)))))))))

(deftest build-rejects-unknown-hard-questions
  (h/with-temp-dir [dir]
    (spit (str dir "/hard.csv") "user_id,question_index\n3,0\n")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"hard-subset"
                          (build/build! {:users-dir (str fixtures "/users") :output-dir (str dir "/data")
                                         :train-users (build/parse-user-range "3-4")
                                         :test-users (build/parse-user-range "7")
                                         :hard-subset (str dir "/hard.csv")})))))

(deftest random-generators-match-python
  (let [ref (util/read-json "test/fixtures/random.json")
        r (pr/random-state 42)]
    (is (= (get ref "random") (vec (repeatedly 3 #(pr/random r)))))
    (is (= (get ref "randint") (vec (repeatedly 5 #(pr/randint r 15 25)))))
    (is (= (get ref "choice") (vec (repeatedly 5 #(str (pr/choice r "ABCDE"))))))
    (is (= (get ref "shuffle") (pr/shuffle r (range 10))))
    (is (= (get ref "bits") [(pr/getrandbits r 40) (pr/getrandbits r 7)]))
    (is (= (get ref "seed_big") (pr/randrange (pr/random-state (+ (bit-shift-left 123456789 8) 1 17)) 4)))
    (doseq [[n p] (get ref "perm")]
      (is (= p (pr/permutation (pr/default-rng 23) (parse-long n))) (str "permutation of " n)))
    (testing "the per-user split is the permutation's tail and head"
      (let [[train held] (build/split-questions 145 0.1 23)]
        (is (= 15 (count held)))
        (is (= (get-in ref ["perm" "145"]) (into held train)))))))

(deftest csv-round-trip
  (let [rows [{"a" "plain" "b" "with, comma"} {"a" "quote \"q\"" "b" "line\nbreak"}]]
    (is (= rows (csv/read-maps (csv/write-rows ["a" "b"] rows))))))

(deftest personas-are-selected-by-range
  (h/with-temp-dir [dir]
    (doseq [id [2 10 11]] (spit (str dir "/user_" id ".txt") "persona"))
    (spit (str dir "/notes.txt") "not a persona")
    (is (= [10 11] (mapv first (gen/find-personas dir 3 nil))))
    (is (= [2 10] (mapv first (gen/find-personas dir nil 10))))))
