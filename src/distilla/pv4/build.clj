(ns distilla.pv4.build
  "Build the PersonalizationV4 train / validation / test parquet files from
  per-user generation outputs (personalizationv4/build_dataset.py).

  Training users contribute a seeded 90/10 split of their questions to train
  and validation; evaluation users contribute every question to test. Each
  row carries the user's full chat history as its documents. The split draws
  numpy's default_rng(23) permutation, so released data rebuilds exactly."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [distilla.cli :as cli]
            [distilla.csv :as csv]
            [distilla.parquet :as pq]
            [distilla.pv4.generate :as gen]
            [distilla.pyrandom :as pr]))

(def splits ["train" "validation" "test"])
(def train-users "6-125")
(def test-users "126-155")
(def validation-fraction 0.1)
(def split-seed 23)
(def user-splits-file "user_splits.csv")
(def user-splits-columns ["user_id" "role" "questions" "train" "validation" "test" "hard"])

(defn parse-user-range
  "An inclusive range such as \"6-125\" or a single id such as \"7\", as a set."
  [spec]
  (let [[start end] (let [[a b] (str/split spec #"-" 2)] [(parse-long a) (parse-long (or b a))])]
    (set (range start (inc end)))))

(defn find-users [users-dir]
  (into {} (keep (fn [p] (when-let [[_ id] (re-matches #"^user_(\d+)$" (str (fs/file-name p)))]
                           (when (fs/directory? p) [(parse-long id) (str p)])))
                 (fs/list-dir users-dir))))

(defn read-questions
  "A user's qa.csv as [{:index :question :choices :answer}], by index."
  [path]
  (->> (csv/read-maps (slurp (str path)))
       (mapv (fn [row]
               (let [choices (mapv #(get row % "") gen/choice-columns)
                     letter (str/upper-case (str/trim (get row "correct_choice" "")))
                     idx (.indexOf ^java.util.List gen/letters letter)]
                 (when (or (neg? idx) (empty? (nth choices idx)))
                   (throw (ex-info (str path ": question " (get row "index") " has no valid correct choice") {})))
                 {:index (parse-long (get row "index")) :question (get row "question")
                  :choices choices :answer (nth choices idx)})))
       (sort-by :index)
       vec))

(defn read-chats
  "A user's chats in topic order."
  [chats-dir]
  (->> (fs/glob chats-dir "*.txt")
       (sort-by #(parse-long (str/replace (str (fs/file-name %)) #"\.txt$" "")))
       (mapv #(slurp (str %)))))

(defn split-questions
  "[train-positions held-out-positions] of one user: numpy's default_rng(seed)
  permutation, its first ceil(fraction * n) held out (datasets'
  train_test_split rule)."
  [n fraction seed]
  (let [perm (pr/permutation (pr/default-rng seed) n)
        held (long (Math/ceil (* fraction n)))]
    [(subvec perm held) (subvec perm 0 held)]))

(defn read-hard-subset [path]
  (if path
    (set (for [row (csv/read-maps (slurp (str path)))]
           [(parse-long (get row "user_id")) (parse-long (get row "question_index"))]))
    #{}))

(defn build!
  "Write {train,validation,test}.parquet and user_splits.csv; returns rows per split."
  [{:keys [users-dir output-dir train-users test-users fraction seed hard-subset id-column]
    :or {fraction validation-fraction seed split-seed id-column "user_id"}}]
  (when (seq (filter train-users test-users))
    (throw (ex-info "Training and evaluation user ranges overlap" {})))
  (let [users (find-users users-dir)
        hard (read-hard-subset hard-subset)
        columns [id-column "question" "answer" "choices" "documents" "hard"]
        acc (reduce
             (fn [acc user-id]
               (if-not (or (train-users user-id) (test-users user-id))
                 acc
                 (let [questions (read-questions (fs/path (users user-id) "qa.csv"))
                       documents (read-chats (fs/path (users user-id) "chats"))
                       parts (if (train-users user-id)
                               (let [[tr held] (split-questions (count questions) fraction seed)]
                                 {"train" (mapv questions tr) "validation" (mapv questions held)})
                               {"test" questions})
                       rows-of (fn [qs] (mapv (fn [q] {id-column user-id "question" (:question q) "answer" (:answer q)
                                                       "choices" (:choices q) "documents" documents
                                                       "hard" (contains? hard [user-id (:index q)])})
                                              qs))]
                   (-> (reduce (fn [acc [split qs]] (update-in acc [:rows split] into (rows-of qs))) acc parts)
                       (update :test-keys into (when (test-users user-id) (map (fn [q] [user-id (:index q)]) questions)))
                       (update :user-rows conj
                               (merge {"user_id" user-id
                                       "role" (if (train-users user-id) "training" "evaluation")
                                       "questions" (count questions)
                                       "hard" (count (filter #(contains? hard [user-id (:index %)]) questions))}
                                      (into {} (for [s splits] [s (count (get parts s []))]))))))))
             {:rows (zipmap splits (repeat [])) :test-keys #{} :user-rows []}
             (sort (keys users)))]
    (fs/create-dirs output-dir)
    (doseq [s splits]
      (pq/write-rows (str (fs/path output-dir (str s ".parquet"))) (get-in acc [:rows s]) columns))
    (spit (str (fs/path output-dir user-splits-file)) (csv/write-rows user-splits-columns (:user-rows acc)))
    (let [unmatched (remove (:test-keys acc) hard)]
      (when (seq unmatched)
        (throw (ex-info (str (count unmatched) " hard-subset entries do not match a test question") {}))))
    (into {} (for [s splits] [s (count (get-in acc [:rows s]))]))))

(def spec
  {:users_dir {:desc "Directory of user_N/ generation outputs." :required true}
   :output_dir {:desc "Directory for {train,validation,test}.parquet and user_splits.csv." :required true}
   :hard_subset {:desc "CSV of (user_id, question_index) hard questions."}
   :train_users {:desc "Inclusive id range of training users." :default train-users}
   :test_users {:desc "Inclusive id range of evaluation users." :default test-users}
   :validation_fraction {:desc "Validation share per training user." :default validation-fraction :coerce :double}
   :seed {:desc "Seed of the per-user split." :default split-seed :coerce :long}
   :id_column {:desc "Name of the user-id column (collection_id for the Memorilla data layout)." :default "user_id"}})

(defn run [opts]
  (let [counts (build! {:users-dir (:users_dir opts) :output-dir (:output_dir opts)
                        :train-users (parse-user-range (:train_users opts))
                        :test-users (parse-user-range (:test_users opts))
                        :fraction (:validation_fraction opts) :seed (:seed opts)
                        :hard-subset (:hard_subset opts) :id-column (:id_column opts)})]
    (doseq [[s n] counts]
      (println (str s ": " n " rows -> " (fs/path (:output_dir opts) (str s ".parquet")))))))

(defn -main [& args]
  (cli/exit-on-error #(run (cli/parse "usage: jolt pv4-build --users_dir DIR --output_dir DIR" spec args)))
  (System/exit 0))
