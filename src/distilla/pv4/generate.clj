(ns distilla.pv4.generate
  "Generate PersonalizationV4 users from persona profiles with an
  OpenAI-compatible API (personalizationv4/generate.py).

  Each user goes through five resumable steps: question categories, chat
  topics, two-turn chats, evaluation questions, and the parsed question table
  with shuffled answer positions. Every output is written atomically and
  existing files are never requested again, so rerunning completes an
  interrupted run."
  (:require [babashka.fs :as fs]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [distilla.cli :as cli]
            [distilla.csv :as csv]
            [distilla.openai :as openai]
            [distilla.pyrandom :as pr]))

(def letters ["A" "B" "C" "D" "E"])
(def choice-columns (mapv #(str "choice_" (str/lower-case %)) letters))
(def qa-columns (into ["index" "category" "question"] (concat choice-columns ["correct_choice" "rationale"])))
(def qa-patterns
  [["question" #"(?s)<QUESTION>\s*(.*?)(?=<CHOICE_A>|$)"]
   ["choice_a" #"(?s)<CHOICE_A>\s*(.*?)(?=<CHOICE_B>|$)"]
   ["choice_b" #"(?s)<CHOICE_B>\s*(.*?)(?=<CHOICE_C>|$)"]
   ["choice_c" #"(?s)<CHOICE_C>\s*(.*?)(?=<CHOICE_D>|$)"]
   ["choice_d" #"(?s)<CHOICE_D>\s*(.*?)(?=<CHOICE_E>|$)"]
   ["choice_e" #"(?s)<CHOICE_E>\s*(.*?)(?=<CORRECT_CHOICE>|$)"]
   ["correct_choice" #"(?s)<CORRECT_CHOICE>\s*([A-E])\b"]
   ["rationale" #"(?s)<RATIONALE[^>]*>\s*(.*?)$"]])
(def persona-file #"^user_(\d+)\.txt$")

(def defaults
  {:model "gpt-5.1" :reasoning-effort "none" :chat-model "gpt-5-mini" :chat-reasoning-effort "minimal"
   :num-categories 10 :num-topics 200 :num-questions 15 :max-workers 50 :max-retries 3 :seed 42})

;; --- prompts -------------------------------------------------------------------------------

(def ^:private templates
  (delay (into {} (for [k [:categories :topics :chat :questions]]
                    [k (slurp (io/resource (str "pv4/" (name k) ".txt")))]))))

(defn- fill
  "str.format with named fields, in one pass (values are not re-scanned)."
  [template values]
  (str/replace template #"\{([a-z_]+)\}" (fn [[whole k]] (str (get values k whole)))))

(defn categories-prompt [persona n] (fill (:categories @templates) {"persona" persona "num_categories" n}))
(defn topics-prompt [persona n] (fill (:topics @templates) {"persona" persona "num_topics" n}))
(defn chat-prompt [persona topic] (fill (:chat @templates) {"persona" persona "topic" topic}))
(defn questions-prompt [persona category n]
  (fill (:questions @templates) {"persona" persona "category" category "num_questions" n}))

;; --- parsing --------------------------------------------------------------------------------

(defn parse-numbered-list
  "The items of a numbered list, blank lines and \"N.\" prefixes dropped."
  [text]
  (->> (str/split (str/trim text) #"\n")
       (map str/trim)
       (remove str/blank?)
       (mapv #(str/replace-first % #"^\d+\.\s*" ""))))

(defn- python-strip [s] (str/trim s))

(defn parse-questions
  "One category's question text as records of its tagged fields; blocks
  without a question are dropped, missing fields are absent."
  [text]
  (vec (for [block (str/split (python-strip text) #"<QUESTION>" -1)
             :when (not (str/blank? block))
             :let [tagged (str "<QUESTION>" block)
                   record (into {} (for [[field pattern] qa-patterns
                                         :let [m (re-find pattern tagged)]
                                         :when m]
                                     [field (python-strip (second m))]))]
             :when (not (str/blank? (get record "question")))]
         record)))

(defn shuffle-choices
  "Move each correct answer to a random letter and shuffle the distractors
  around it, with Python's random.Random so positions match the released
  data. Records without a valid correct letter or correct text are dropped."
  [records rng]
  (vec (keep (fn [record]
               (let [m (re-find #"(?i)^[A-E]" (get record "correct_choice" ""))]
                 (when m
                   (let [correct (get record (str "choice_" (str/lower-case m)) "")]
                     (when-not (empty? correct)
                       (let [new-letter (pr/choice rng letters)
                             choices (mapv #(get record % "") choice-columns)
                             idx (.indexOf ^java.util.List choices correct)
                             others (into (subvec choices 0 idx) (subvec choices (inc idx)))
                             shuffled (pr/shuffle rng others)
                             placed (let [at (.indexOf ^java.util.List letters new-letter)]
                                      (into (conj (subvec shuffled 0 at) correct) (subvec shuffled at)))]
                         (merge record (zipmap choice-columns placed) {"correct_choice" new-letter})))))))
             records)))

(defn build-question-table
  "Parse every category file of a user and shuffle answer positions."
  [qa-dir num-categories seed]
  (let [records (vec (mapcat (fn [category]
                               (map #(assoc % "category" category)
                                    (parse-questions (slurp (str (fs/path qa-dir (str category ".txt")))))))
                             (range num-categories)))
        rows (shuffle-choices records (pr/random-state seed))]
    (vec (map-indexed (fn [i row] (assoc (into {} (for [c qa-columns] [c (get row c "")])) "index" i)) rows))))

(defn write-text!
  "Write a file atomically (through a .tmp sibling)."
  [path text]
  (fs/create-dirs (fs/parent path))
  (let [tmp (str path ".tmp")]
    (spit tmp text)
    (fs/move tmp path {:replace-existing true})))

;; --- one user ---------------------------------------------------------------------------------

(defn- generate-list [client config path prompt]
  (when-not (fs/exists? path)
    (write-text! path (str/join "\n" (parse-numbered-list
                                      (openai/complete client prompt (:model config) (:reasoning-effort config))))))
  (str/split (slurp (str path)) #"\n" -1))

(defn- generate-files
  "One file per prompt, skipping files that exist; true when all exist after."
  [client jobs model effort strip?]
  (let [pending (vec (remove (comp fs/exists? first) jobs))
        results (openai/complete-all client (for [[_ prompt] pending] [prompt model effort]))]
    (doseq [[[path _] {:keys [text error]}] (map vector pending results)]
      (if text
        (write-text! path (if strip? (str/trim text) text))
        (binding [*out* *err*] (println (str path ": " error)))))
    (every? (comp fs/exists? first) jobs)))

(defn run-user!
  "Every output of one user under user-dir; true when qa.csv is written."
  [client config persona-path user-dir]
  (let [persona (slurp (str persona-path))]
    (try
      (let [categories (generate-list client config (fs/path user-dir "categories.txt")
                                      (categories-prompt persona (:num-categories config)))
            topics (generate-list client config (fs/path user-dir "chat_topics.txt")
                                  (topics-prompt persona (:num-topics config)))
            chats-done (generate-files client
                                       (map-indexed (fn [i topic] [(str (fs/path user-dir "chats" (str i ".txt")))
                                                                   (chat-prompt persona topic)])
                                                    topics)
                                       (:chat-model config) (:chat-reasoning-effort config) true)
            questions-done (generate-files client
                                           (map-indexed (fn [i category] [(str (fs/path user-dir "qa" (str i ".txt")))
                                                                          (questions-prompt persona category
                                                                                            (:num-questions config))])
                                                        categories)
                                           (:model config) (:reasoning-effort config) false)]
        (if-not (and chats-done questions-done)
          (do (println (str (fs/file-name user-dir) " is incomplete; rerun to fill in the missing chats or questions"))
              false)
          (let [table (fs/path user-dir "qa.csv")]
            (when-not (fs/exists? table)
              (let [rows (build-question-table (fs/path user-dir "qa") (count categories) (:seed config))]
                (write-text! table (csv/write-rows qa-columns rows))
                (println (str (fs/file-name user-dir) ": " (count rows) " questions"))))
            true)))
      (catch clojure.lang.ExceptionInfo e
        (println (str (fs/file-name user-dir) ": " (ex-message e)))
        false))))

(defn find-personas
  "[user-id path] of user_N.txt files, ascending, within an inclusive range."
  [persona-dir user-start user-end]
  (->> (fs/list-dir persona-dir)
       (keep (fn [p] (when-let [[_ id] (re-matches persona-file (str (fs/file-name p)))]
                       [(parse-long id) (str p)])))
       (filter (fn [[id]] (and (or (nil? user-start) (>= id user-start)) (or (nil? user-end) (<= id user-end)))))
       (sort-by first)
       vec))

(def spec
  {:persona_dir {:desc "Directory of user_N.txt persona profiles." :required true}
   :output_dir {:desc "Directory that receives one user_N/ per user." :required true}
   :user_start {:desc "Smallest user id to generate (inclusive)." :coerce :long}
   :user_end {:desc "Largest user id to generate (inclusive)." :coerce :long}
   :model {:desc "Model for categories, topics and questions." :default (:model defaults)}
   :reasoning_effort {:desc "Reasoning effort for --model." :default (:reasoning-effort defaults)}
   :chat_model {:desc "Model for the two-turn chats." :default (:chat-model defaults)}
   :chat_reasoning_effort {:desc "Reasoning effort for --chat_model." :default (:chat-reasoning-effort defaults)}
   :num_categories {:desc "Categories per user." :default (:num-categories defaults) :coerce :long}
   :num_topics {:desc "Chat topics (and chats) per user." :default (:num-topics defaults) :coerce :long}
   :num_questions {:desc "Questions per category." :default (:num-questions defaults) :coerce :long}
   :max_workers {:desc "Concurrent requests." :default (:max-workers defaults) :coerce :long}
   :max_retries {:desc "Attempts per request." :default (:max-retries defaults) :coerce :long}
   :seed {:desc "Seed of the answer-position shuffle." :default (:seed defaults) :coerce :long}
   :api {:desc "responses (OpenAI Responses API) or chat (chat/completions, e.g. llama-server)." :default "responses"}
   :base_url {:desc "API base URL (default $OPENAI_BASE_URL or https://api.openai.com/v1)."}})

(defn run [opts]
  (let [config {:model (:model opts) :reasoning-effort (:reasoning_effort opts)
                :chat-model (:chat_model opts) :chat-reasoning-effort (:chat_reasoning_effort opts)
                :num-categories (:num_categories opts) :num-topics (:num_topics opts)
                :num-questions (:num_questions opts) :seed (:seed opts)}
        client (openai/client (cond-> {:api (keyword (:api opts)) :max-workers (:max_workers opts)
                                       :max-retries (:max_retries opts)}
                                (:base_url opts) (assoc :base-url (:base_url opts))))
        personas (find-personas (:persona_dir opts) (:user_start opts) (:user_end opts))]
    (when (empty? personas)
      (throw (ex-info (str "No user_N.txt persona files selected in " (:persona_dir opts)) {})))
    (let [incomplete (vec (for [[id path] personas
                                :let [_ (println (str "Generating user_" id))]
                                :when (not (run-user! client config path (fs/path (:output_dir opts) (str "user_" id))))]
                            id))]
      (if (seq incomplete)
        (println "Incomplete users (rerun the same command to resume):" incomplete)
        (println (str "Generated " (count personas) " users in " (:output_dir opts)))))))

(defn -main [& args]
  (cli/exit-on-error #(run (cli/parse "usage: jolt pv4-generate --persona_dir DIR --output_dir DIR" spec args)))
  (System/exit 0))
