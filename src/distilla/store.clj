(ns distilla.store
  "Precomputed document and question embeddings on disk: the reader
  (`open-store`) and the writer (`writer`), in the layout of the Python
  implementation, so the published embeddings are read as they are.

  One (config, split) lives in embeddings/<encoder>/<config>/<split>/ under
  the data root:

  - index.safetensors: collection_ids int64 [C] (ascending), shard int32 [C],
    start int64 [C] and length int64 [C] locating each collection's rows.
  - documents-XXXXX.safetensors: tensor embeddings float16 [n_i, D];
    collections never span two shards.
  - questions.safetensors: tensor embeddings float16 [R, D], row-aligned with
    data/<config>/<split>."
  (:require [babashka.fs :as fs]
            [distilla.paths :as paths]
            [distilla.safetensors :as st]
            [distilla.tensor :as t]))

(def index-file "index.safetensors")
(def questions-file "questions.safetensors")
(def tensor-name "embeddings")
(def max-shard-bytes (* 4 1024 1024 1024))
(def lock-file ".distilla.lock")

(defn shard-file [shard] (format "documents-%05d.safetensors" shard))

(defn embedding-dir
  "root/embeddings/<encoder slug>/<config>/<split>."
  ([config split root] (embedding-dir config split root paths/default-encoder))
  ([config split root encoder]
   (str (fs/path root "embeddings" (paths/encoder-slug encoder) config split))))

(defn missing-files
  "The files of one split's embeddings that are absent: the index, or the
  question file and the document shards the index refers to."
  [directory]
  (let [index (fs/path directory index-file)]
    (if-not (fs/exists? index)
      [index-file]
      (let [shards (->> (:data (st/read-tensor (st/open index) "shard")) seq distinct sort)
            names (into [questions-file] (map shard-file shards))]
        (vec (remove #(fs/exists? (fs/path directory %)) names))))))

;; --- reader ----------------------------------------------------------------------

(defn open-store
  "Open one split's embeddings. Throws when they are absent or incomplete."
  ([config split] (open-store config split (paths/data-root) paths/default-encoder))
  ([config split root] (open-store config split root paths/default-encoder))
  ([config split root encoder]
   (let [directory (embedding-dir config split (or root (paths/data-root)) encoder)
         missing (missing-files directory)]
     (when (seq missing)
       (throw (ex-info (str "Embeddings of " config "/" split " in " directory " lack " (pr-str missing) ".")
                       {:type :file-not-found :missing missing :directory directory})))
     (let [index (st/open (fs/path directory index-file))
           col #(:data (st/read-tensor index %))]
       {:config config :split split :directory directory
        :collection-ids (col "collection_ids")
        :shard (col "shard") :start (col "start") :length (col "length")
        :handles (atom {})}))))

(defn store-count [store] (alength ^longs (:collection-ids store)))

(defn- position
  "The index position of a collection id, or nil."
  [store collection-id]
  (let [^longs ids (:collection-ids store)
        target (long collection-id)]
    (loop [lo 0 hi (dec (alength ids))]
      (when (<= lo hi)
        (let [mid (quot (+ lo hi) 2)
              v (aget ids mid)]
          (cond (< v target) (recur (inc mid) hi)
                (> v target) (recur lo (dec mid))
                :else mid))))))

(defn contains-collection? [store collection-id] (some? (position store collection-id)))

(defn- handle [store filename]
  (or (get @(:handles store) filename)
      (let [h (st/open (fs/path (:directory store) filename))]
        (swap! (:handles store) assoc filename h)
        h)))

(defn num-questions [store]
  (first (st/shape (handle store questions-file) tensor-name)))

(defn dim [store]
  (second (st/shape (handle store questions-file) tensor-name)))

(defn documents
  "A collection's document embeddings: an [n x D] matrix."
  [store collection-id]
  (let [pos (or (position store collection-id)
                (throw (ex-info (str "Collection " collection-id " not found in " (:config store) "/" (:split store) ".")
                                {:type :key-error :collection-id collection-id})))
        shard (aget ^longs (:shard store) pos)
        start (aget ^longs (:start store) pos)
        length (aget ^longs (:length store) pos)
        h (handle store (shard-file shard))
        cols (second (st/shape h tensor-name))]
    (if (zero? length)
      (t/matrix 0 cols)
      (st/read-matrix h tensor-name start (+ start length)))))

(defn question
  "The question embedding of one row of the split, a float array."
  [store row]
  (:data (st/read-matrix (handle store questions-file) tensor-name row (inc row))))

(defn questions
  "Every question embedding of the split, an [R x D] matrix."
  [store]
  (st/read-matrix (handle store questions-file) tensor-name))

;; --- writer -----------------------------------------------------------------------

(defn writer
  "A writer for one split. Add collections in strictly ascending id order with
  add!, then write-questions! and close!. A new shard starts whenever the next
  non-empty collection would push the current one past max-shard-bytes; a
  collection larger than that gets a shard of its own. Collections without
  documents are indexed in the current shard."
  ([directory] (writer directory max-shard-bytes))
  ([directory max-bytes]
   (fs/create-dirs directory)
   (atom {:directory (str directory) :max-shard-bytes max-bytes
          :ids [] :shards [] :starts [] :lengths []
          :buffer [] :buffer-rows 0 :buffer-bytes 0 :shard-index 0 :dim nil})))

(defn- flush! [w]
  (let [{:keys [buffer directory shard-index dim]} @w]
    (when (seq buffer)
      (let [m (t/concat-rows buffer dim)]
        (st/save-file {tensor-name {:dtype "F16" :shape [(:rows m) dim] :data (:data m)}}
                      (str (fs/path directory (shard-file shard-index)))))
      (swap! w assoc :buffer [] :buffer-rows 0 :buffer-bytes 0 :shard-index (inc shard-index)))))

(defn add!
  "Append one collection's document embeddings (a matrix, stored as float16)."
  [w collection-id m]
  (let [{:keys [ids]} @w]
    (when (and (seq ids) (<= collection-id (peek ids)))
      (throw (ex-info (str "Collection ids must be strictly ascending: " collection-id " after " (peek ids) ".")
                      {:type :value-error}))))
  (let [rows (:rows m)
        size (* 2 rows (:cols m))
        {:keys [buffer buffer-bytes max-shard-bytes]} @w]
    (when (and (pos? size) (seq buffer) (> (+ buffer-bytes size) max-shard-bytes))
      (flush! w))
    (swap! w (fn [s]
               (cond-> (-> s
                           (update :ids conj (long collection-id))
                           (update :shards conj (:shard-index s))
                           (update :starts conj (:buffer-rows s))
                           (update :lengths conj rows))
                 (pos? rows) (-> (update :buffer conj m)
                                 (update :buffer-rows + rows)
                                 (update :buffer-bytes + size)
                                 (assoc :dim (:cols m))))))))

(defn write-questions!
  "Write the row-aligned question embeddings (a matrix, stored as float16)."
  [w m]
  (st/save-file {tensor-name {:dtype "F16" :shape [(:rows m) (:cols m)] :data (:data m)}}
                (str (fs/path (:directory @w) questions-file))))

(defn close!
  "Write the last shard and the index."
  [w]
  (flush! w)
  (let [{:keys [ids shards starts lengths directory]} @w]
    (st/save-file {"collection_ids" {:dtype "I64" :shape [(count ids)] :data ids}
                   "shard" {:dtype "I32" :shape [(count shards)] :data shards}
                   "start" {:dtype "I64" :shape [(count starts)] :data starts}
                   "length" {:dtype "I64" :shape [(count lengths)] :data lengths}}
                  (str (fs/path directory index-file)))))

(defmacro with-writer [[sym directory & [max-bytes]] & body]
  `(let [~sym (writer ~directory ~@(when max-bytes [max-bytes]))
         result# (do ~@body)]
     (close! ~sym)
     result#))
