(ns distilla.data-test
  "Data access, chat formatting, collation, and the embedding store."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [distilla.data :as data]
            [distilla.dataset :as ds]
            [distilla.embeddings :as emb]
            [distilla.helpers :as h]
            [distilla.safetensors :as st]
            [distilla.store :as store]
            [distilla.tensor :as t]
            [distilla.tokenizer :as tok]
            [distilla.util :as util]))

(def num-memories 3)
(def system-prompt "You are a helpful assistant")
(def suffix " please answer briefly")

;; --- messages and data access ---------------------------------------------------------------

(deftest messages-and-placeholder
  (is (= (str "\nMemory" util/memory-token util/memory-token "Memory") (data/memory-placeholder 2)))
  (is (= [{:role "system" :content (str "System" (data/memory-placeholder 2))}
          {:role "user" :content "who wrote hamlet ok"}
          {:role "assistant" :content "shakespeare"}]
         (data/build-messages "  who wrote hamlet " "System" 2 " ok" " shakespeare ")))
  (is (= 2 (count (data/build-messages "q" "s" 1)))))

(deftest local-repository-access
  (h/with-temp-dir [root]
    (h/data-repo root (h/fake-encoder))
    (is (= root (data/ensure-data "toy" ["train" "validation"] root root)))
    (is (= [] (data/split-files "toy" "validation" root)))
    (let [dataset (data/load-split "toy" "train" root ["collection_id" "question" "missing"])]
      (is (= ["collection_id" "question"] (ds/column-names dataset)))
      (is (= (mapv #(get % "collection_id") (h/make-rows)) (ds/column dataset "collection_id"))))
    (is (thrown? clojure.lang.ExceptionInfo (data/load-split "toy" "validation" root)))))

(deftest split-completeness
  (h/with-temp-dir [root]
    (let [rows (h/make-rows)]
      (h/write-split root "toy" "train" rows)
      (is (data/split-complete? "toy" "train" root))
      (is (not (data/split-complete? "toy" "validation" root)))
      (let [dir (fs/path root "data" "toy")]
        (fs/move (fs/path dir "train-00000-of-00001.parquet") (fs/path dir "train-00000-of-00002.parquet"))
        (is (not (data/split-complete? "toy" "train" root)))
        (is (= root (data/ensure-data "toy" ["train"] root root)))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"missing" (data/load-split "toy" "train" root)))
        (fs/move (fs/path dir "train-00000-of-00002.parquet") (fs/path dir "train-part.parquet"))
        (is (data/split-complete? "toy" "train" root))
        (is (= (count rows) (ds/row-count (data/load-split "toy" "train" root))))))))

(deftest memory-dataset-tracks-sources
  (h/with-temp-dir [root]
    (h/data-repo root (h/fake-encoder))
    (let [train (data/load-split "toy" "train" root)
          test (data/load-split "toy" "test" root)
          mds (data/memory-dataset [["toy" "train" train] ["toy" "test" test]])
          item (data/dataset-item mds (inc (ds/row-count train)))]
      (is (= (+ (ds/row-count train) (ds/row-count test)) (data/dataset-size mds)))
      (is (= ["toy" "test" 1] [(item "config") (item "split") (item "row")]))
      (is (= (get (ds/row test 1) "question") (item "question"))))))

;; --- collation ------------------------------------------------------------------------------------

(defn- collator [tokenizer root train & {:keys [suffix max-length] :or {suffix "" max-length 128}}]
  (let [st (store/open-store "toy" "train" root)
        mds (data/memory-dataset [["toy" "train" (data/load-split "toy" "train" root)]])]
    [{:tokenizer tokenizer :stores {["toy" "train"] st} :num-memories num-memories :max-length max-length
      :system-prompt system-prompt :suffix suffix :train train}
     (mapv #(data/dataset-item mds %) (range (data/dataset-size mds)))
     st]))

(deftest training-batch
  (h/with-temp-dir [root]
    (h/data-repo root (h/fake-encoder))
    (let [tokenizer (h/word-tokenizer)
          mem-id (tok/memory-token-id tokenizer)
          [opts items st] (collator tokenizer root true)
          {:keys [input-ids attention-mask labels doc-embeds doc-padding-mask question-embeds]} (data/collate opts items)
          width (count (first input-ids))
          lengths (map #(reduce + %) attention-mask)]
      (is (every? #(= num-memories (count (filter #{mem-id} %))) input-ids))
      (is (= width (apply max lengths)))
      (is (< (apply min lengths) width))
      (doseq [row (range (count items))]
        (let [padding (- width (nth lengths row))]
          (is (every? zero? (take padding (nth attention-mask row))))
          (is (every? #{1} (drop padding (nth attention-mask row))))
          (is (every? #{(tok/pad-id tokenizer)} (take padding (nth input-ids row))))
          (is (= (tok/encode tokenizer (str (get (nth items row) "answer") "<|im_end|>"))
                 (vec (remove #{data/ignore-index} (nth labels row)))))
          (is (every? #{data/ignore-index} (take padding (nth labels row))))))
      (doseq [[row item] (map-indexed vector items)]
        (is (= (vec (store/question st row)) (vec (nth question-embeds row))))
        (let [documents (store/documents st (get item "collection_id"))
              n (:rows documents)
              padded (nth doc-embeds row)]
          (is (= (vec (:data documents)) (vec (:data (t/take-rows padded n)))))
          (is (not-any? true? (take n (nth doc-padding-mask row))))
          (is (every? true? (drop n (nth doc-padding-mask row))))
          (is (every? zero? (drop (* n (:cols padded)) (:data padded)))))))))

(deftest generation-batch-with-suffix
  (h/with-temp-dir [root]
    (h/data-repo root (h/fake-encoder))
    (let [tokenizer (h/word-tokenizer)
          [opts items] (collator tokenizer root false :suffix suffix)
          [plain] (collator tokenizer root false)
          batch (data/collate opts items)
          reference (data/collate plain items)
          header (tok/encode tokenizer "<|im_start|> assistant")]
      (is (not (contains? batch :labels)))
      (doseq [row (range (count items))]
        (let [ids (vec (keep-indexed (fn [i id] (when (pos? (nth (nth (:attention-mask batch) row) i)) id))
                                     (nth (:input-ids batch) row)))]
          (is (= header (subvec ids (- (count ids) (count header)))))
          (is (= 1 (count (re-seq #"please answer briefly" (tok/decode tokenizer ids)))))))
      (is (= (map vec (:question-embeds batch)) (map vec (:question-embeds reference)))))))

(deftest truncation-respects-max-length
  (h/with-temp-dir [root]
    (h/data-repo root (h/fake-encoder))
    (let [[opts items] (collator (h/word-tokenizer) root true :max-length 12)
          batch (data/collate opts (subvec items 0 2))]
      (is (= 12 (count (first (:input-ids batch))))))))

;; --- embedding store -------------------------------------------------------------------------------

(def dim 4)

(defn- collections []
  (into (sorted-map)
        (for [[id size] {2 3 5 1 9 0 11 4 20 12 21 2}]
          [id (t/matrix size dim (t/round-to-half (h/normals id (* size dim))))])))

(defn- write-store [dir colls max-rows]
  (let [questions (t/matrix 5 dim (t/round-to-half (h/normals 99 (* 5 dim))))]
    (store/with-writer [w dir (* max-rows dim 2)]
      (doseq [[id m] colls] (store/add! w id m))
      (store/write-questions! w questions))
    questions))

(deftest store-round-trip-across-shards
  (h/with-temp-dir [root]
    (let [colls (collections)
          dir (store/embedding-dir "toy" "train" root)
          questions (write-store dir colls 5)
          st (store/open-store "toy" "train" root)]
      (is (> (count (fs/glob dir "documents-*.safetensors")) 2))
      (is (= (count colls) (store/store-count st)))
      (is (= 5 (store/num-questions st)))
      (doseq [[id m] colls]
        (is (store/contains-collection? st id))
        (let [actual (store/documents st id)]
          (is (= [(:rows m) dim] [(:rows actual) (:cols actual)]))
          (is (= (vec (:data m)) (vec (:data actual))))))
      (is (= (vec (:data questions)) (vec (:data (store/questions st)))))
      (doseq [row (range 5)]
        (is (= (vec (t/row questions row)) (vec (store/question st row))))))))

(deftest collections-stay-within-one-shard
  (h/with-temp-dir [dir]
    (write-store dir (collections) 5)
    (let [index (st/open (fs/path dir store/index-file))
          col #(vec (:data (st/read-tensor index %)))
          ids (col "collection_ids") shards (col "shard") starts (col "start") lengths (col "length")]
      (is (= ["I64" "I32" "I64" "I64"] (map #(:dtype (st/info index %)) ["collection_ids" "shard" "start" "length"])))
      (is (= (vec (keys (collections))) ids))
      (doseq [shard (distinct shards)]
        (let [rows (first (st/shape (st/open (fs/path dir (store/shard-file shard))) store/tensor-name))
              members (filter #(= shard (nth shards %)) (range (count ids)))]
          (is (= rows (apply max (map #(+ (nth starts %) (nth lengths %)) members))))
          (when (> (count members) 1) (is (<= rows 5)))))
      (let [oversized (.indexOf ids 20)]
        (is (= 1 (count (filter #{(nth shards oversized)} shards))))))))

(deftest empty-collections
  (h/with-temp-dir [root]
    (store/with-writer [w (store/embedding-dir "toy" "train" root) (* 2 dim 2)]
      (store/add! w 1 (t/matrix 5 dim (float-array (repeat (* 5 dim) 1.0))))
      (store/add! w 2 (t/matrix 0 dim))
      (store/write-questions! w (t/matrix 1 dim)))
    (let [st (store/open-store "toy" "train" root)]
      (is (= 0 (:rows (store/documents st 2))))
      (is (= (repeat (* 5 dim) 1.0) (map double (:data (store/documents st 1))))))))

(deftest writer-requires-ascending-ids
  (h/with-temp-dir [dir]
    (let [w (store/writer dir)]
      (store/add! w 5 (t/matrix 1 dim))
      (is (thrown? clojure.lang.ExceptionInfo (store/add! w 5 (t/matrix 1 dim))))
      (is (thrown? clojure.lang.ExceptionInfo (store/add! w 3 (t/matrix 1 dim)))))))

(deftest store-errors
  (h/with-temp-dir [root]
    (is (thrown? clojure.lang.ExceptionInfo (store/open-store "toy" "missing" root)))
    (write-store (store/embedding-dir "toy" "train" root) (collections) 100)
    (let [st (store/open-store "toy" "train" root)]
      (is (thrown? clojure.lang.ExceptionInfo (store/documents st 4)))
      (is (not (store/contains-collection? st 4))))))

(deftest missing-files-are-detected
  (h/with-temp-dir [root]
    (let [dir (store/embedding-dir "toy" "train" root)]
      (is (= [store/index-file] (store/missing-files dir)))
      (write-store dir (collections) 5)
      (is (= [] (store/missing-files dir)))
      (fs/delete (fs/path dir (store/shard-file 1)))
      (is (= [(store/shard-file 1)] (store/missing-files dir)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"documents-00001" (store/open-store "toy" "train" root))))))

;; --- building embeddings ------------------------------------------------------------------------------

(deftest build-embeddings-matches-layout
  (h/with-temp-dir [root]
    (let [rows (h/make-rows)
          enc (h/fake-encoder)
          dataset (h/write-split root "toy" "train" rows)
          out (store/embedding-dir "toy" "train" root)]
      (emb/build-embeddings dataset out enc {:chunk-size 3})
      (let [st (store/open-store "toy" "train" root)]
        (is (= [3 7] (vec (:collection-ids st))))
        (doseq [id [3 7]]
          (let [docs (some #(when (= id (get % "collection_id")) (get % "documents")) rows)]
            (is (= (mapcat #(vec (h/vector-of enc %)) docs) (vec (:data (store/documents st id)))))))
        (let [qs (store/questions st)]
          (is (= [(count rows) h/embedding-dim] [(:rows qs) (:cols qs)]))
          (doseq [[row item] (map-indexed vector rows)]
            (is (= (vec (h/vector-of enc (str/trim (get item "question")))) (vec (t/row qs row))))))))))

(deftest ensure-embeddings-builds-missing-splits
  (h/with-temp-dir [root]
    (let [enc (h/fake-encoder)]
      (h/write-split root "toy" "train" (h/make-rows))
      (h/write-split root "toy" "test" (subvec (h/make-rows) 0 1) enc)
      (let [test-index (fs/path (store/embedding-dir "toy" "test" root) store/index-file)
            before (fs/last-modified-time test-index)
            result (emb/ensure-embeddings "toy" ["train" "test"] {:repo root :make-encoder (constantly enc)})]
        (is (= root result))
        (is (= before (fs/last-modified-time test-index)))
        (let [st (store/open-store "toy" "train" root)]
          (is (= 2 (store/store-count st)))
          (is (= (ds/row-count (data/load-split "toy" "train" root)) (store/num-questions st))))
        (let [calls (count @(:calls enc))]
          (emb/ensure-embeddings "toy" ["train" "test"] {:repo root :make-encoder (constantly enc)})
          (is (= calls (count @(:calls enc)))))))))

(deftest ensure-embeddings-rejects-absent-splits
  (h/with-temp-dir [root]
    (h/write-split root "toy" "train" (h/make-rows))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"toy splits \[\"validation\"\]"
                          (emb/ensure-embeddings "toy" ["train" "validation"]
                                                 {:repo root
                                                  :make-encoder #(throw (AssertionError. "the encoder was started"))})))))

(deftest build-embeddings-rejects-empty-collections
  (h/with-temp-dir [root]
    (let [enc (h/fake-encoder)
          dataset (h/write-split root "toy" "train" (assoc-in (h/make-rows) [1 "documents"] []))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"\[3\]"
                            (emb/build-embeddings dataset (store/embedding-dir "toy" "train" root) enc)))
      (is (= [] @(:calls enc))))))

(deftest split-rows-must-match-question-embeddings
  (h/with-temp-dir [root]
    (h/data-repo root (h/fake-encoder))
    (let [[dataset st] (emb/load-split-with-embeddings "toy" "train" root)]
      (is (= (ds/row-count dataset) (store/num-questions st) (count (h/make-rows)))))
    (h/write-split root "toy" "train" (subvec (h/make-rows) 0 2))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"2 rows but 3 question embeddings"
                          (emb/load-split-with-embeddings "toy" "train" root)))))
