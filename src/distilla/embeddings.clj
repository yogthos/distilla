(ns distilla.embeddings
  "Building and provisioning the precomputed embeddings: download them from
  the data repository when it publishes them, otherwise compute them with an
  encoder (see distilla.store for the layout)."
  (:require [clojure.string :as str]
            [distilla.data :as data]
            [distilla.dataset :as ds]
            [distilla.encoder :as encoder]
            [distilla.hub :as hub]
            [distilla.paths :as paths]
            [distilla.store :as store]
            [distilla.tensor :as t]
            [babashka.fs :as fs]))

(defn chunk-collections
  "Group consecutive collections so that each group holds about chunk-size
  documents: [begin end) index ranges."
  [lengths chunk-size]
  (loop [index 0 begin 0 total 0 acc []]
    (if (= index (count lengths))
      (if (< begin (count lengths)) (conj acc [begin (count lengths)]) acc)
      (let [total (+ total (nth lengths index))]
        (if (>= total chunk-size)
          (recur (inc index) (inc index) 0 (conj acc [begin (inc index)]))
          (recur (inc index) begin total acc))))))

(defn build-embeddings
  "Embed one split's documents and questions and write them in the store
  layout. Each collection is embedded once, from the first row carrying it;
  questions are embedded stripped, one per row (each distinct text once)."
  ([dataset out-dir enc] (build-embeddings dataset out-dir enc {}))
  ([dataset out-dir enc {:keys [chunk-size max-shard-bytes]
                         :or {chunk-size encoder/embed-chunk-size max-shard-bytes store/max-shard-bytes}}]
   (let [ids (ds/column dataset data/collection-column)
         docs (ds/column dataset data/documents-column)
         first-rows (reduce (fn [m [i id]] (if (contains? m id) m (assoc m id i))) {} (map-indexed vector ids))
         collection-ids (vec (sort (keys first-rows)))
         lengths (mapv #(count (nth docs (first-rows %))) collection-ids)
         empty (vec (keep-indexed (fn [i n] (when (zero? n) (nth collection-ids i))) lengths))]
     (when (seq empty)
       (throw (ex-info (str "Collections without documents: " (pr-str (vec (take 10 empty)))
                            " (" (count empty) " in total).")
                       {:type :value-error :collections empty})))
     (store/with-writer [w out-dir max-shard-bytes]
       (doseq [[begin end] (chunk-collections lengths chunk-size)]
         (let [group (subvec collection-ids begin end)
               texts (vec (for [id group doc (nth docs (first-rows id))] (str doc)))
               vectors (encoder/embed-texts enc texts chunk-size)]
           (loop [ids group offset 0]
             (when-let [id (first ids)]
               (let [n (count (nth docs (first-rows id)))]
                 (store/add! w id (t/select-rows vectors (range offset (+ offset n))))
                 (recur (rest ids) (+ offset n)))))))
       (let [questions (mapv #(str/trim (str %)) (ds/column dataset data/question-column))
             unique (vec (distinct questions))
             position (zipmap unique (range))
             vectors (encoder/embed-texts enc unique chunk-size)]
         (store/write-questions! w (t/select-rows vectors (map position questions))))))))

(defn compute-embeddings
  "Embed the given splits of a config with an encoder from make-encoder."
  [config splits root encoder-id make-encoder]
  (let [enc (make-encoder)]
    (try
      (doseq [split splits]
        (let [dataset (data/load-split config split root [data/collection-column data/documents-column
                                                          data/question-column])]
          (build-embeddings dataset (store/embedding-dir config split root encoder-id) enc)))
      (finally
        (when (instance? java.io.Closeable enc) (.close ^java.io.Closeable enc))))))

(defn ensure-embeddings
  "Make the embeddings of the given splits available locally. Missing splits
  are downloaded from repo when it publishes them; the others are computed
  with the encoder from make-encoder (a fn of no arguments; required only
  when something must be computed). Returns the data root.

  opts: :repo :data-dir :encoder-id (names the embedding folder) :make-encoder"
  [config splits {:keys [repo data-dir encoder-id make-encoder]
                  :or {repo paths/data-repo data-dir paths/data-dir encoder-id paths/default-encoder}}]
  (let [root (paths/data-root repo data-dir)
        _ (fs/create-dirs root)
        splits (vec splits)
        missing (fn [] (filterv #(seq (store/missing-files (store/embedding-dir config % root encoder-id))) splits))]
    (when (seq (missing))
      (when (and (nil? (paths/local-repo repo)) (not (hub/offline?)))
        (hub/snapshot-download {:repo repo :repo-type :dataset :local-dir root
                                :patterns (for [s (missing)]
                                            (str "embeddings/" (paths/encoder-slug encoder-id) "/" config "/" s "/*"))}))
      (when (seq (missing))
        (data/ensure-data config (missing) repo data-dir)
        (let [absent (remove #(data/split-complete? config % root) (missing))]
          (when (seq absent)
            (throw (ex-info (str "No complete parquet files for " config " splits " (pr-str (vec absent))
                                 " under " (fs/path root "data" config) ".")
                            {:type :file-not-found :splits (vec absent)}))))
        (when-not make-encoder
          (throw (ex-info (str "Embeddings of " config " " (pr-str (missing))
                               " must be computed: pass an encoder (--encoder model.gguf).")
                          {:splits (missing)})))
        (compute-embeddings config (missing) root encoder-id make-encoder))
      (when (seq (missing))
        (throw (ex-info (str "Embeddings of " config " " (pr-str (missing)) " are still incomplete under " root ".")
                        {:splits (missing)}))))
    root))

(defn load-split-with-embeddings
  "A split and its embedding store, checked to line up row by row: [dataset store]."
  ([config split root] (load-split-with-embeddings config split root nil paths/default-encoder))
  ([config split root columns encoder-id]
   (let [dataset (data/load-split config split root columns)
         st (store/open-store config split root encoder-id)]
     (when-not (= (ds/row-count dataset) (store/num-questions st))
       (throw (ex-info (str config "/" split " has " (ds/row-count dataset) " rows but " (store/num-questions st)
                            " question embeddings; remove " (:directory st)
                            " to recompute them, or restore the missing data files.")
                       {:type :value-error})))
     [dataset st])))
