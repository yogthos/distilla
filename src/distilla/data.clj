(ns distilla.data
  "Dataset access, chat formatting and batch collation.

  Every config of the data repository stores
  data/<config>/<split>-XXXXX-of-YYYYY.parquet with the columns
  collection_id, question, answer and documents (plus choices and hard where
  they apply)."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [distilla.benchmarks :as b]
            [distilla.dataset :as ds]
            [distilla.hub :as hub]
            [distilla.parquet :as pq]
            [distilla.paths :as paths]
            [distilla.store :as store]
            [distilla.tensor :as t]
            [distilla.tokenizer :as tok]
            [distilla.util :as util]))

(def collection-column "collection_id")
(def question-column "question")
(def answer-column "answer")
(def documents-column "documents")
(def choices-column "choices")
(def hard-column "hard")
(def eval-columns [collection-column question-column answer-column choices-column hard-column])
(def train-split "train")
(def validation-split "validation")
(def test-split "test")
(def shard-name #"-(\d+)-of-(\d+)\.parquet$")
(def default-system-prompt b/helpful-prompt)
(def ignore-index -100)

(defn split-files
  "The sorted parquet files of one split (empty when it is not there)."
  [config split root]
  (let [dir (fs/path root "data" config)]
    (if (fs/directory? dir)
      (->> (fs/glob dir (str split "-*.parquet"))
           (map str)
           sort
           vec)
      [])))

(defn split-complete?
  "Whether every parquet file of a split is present. Files named
  <split>-XXXXX-of-YYYYY.parquet must cover all YYYYY shards; a split stored
  under other names is complete as soon as one file exists."
  [config split root]
  (let [files (split-files config split root)
        shards (keep #(re-find shard-name (str (fs/file-name %))) files)]
    (if (empty? shards)
      (boolean (seq files))
      (let [totals (set (map #(parse-long (nth % 2)) shards))]
        (and (= 1 (count totals))
             (= (set (map #(parse-long (nth % 1)) shards)) (set (range (first totals)))))))))

(defn ensure-data
  "Make the parquet files of the given splits available locally, downloading
  only what is missing from a Hub repo. Returns the data root."
  ([config splits] (ensure-data config splits paths/data-repo paths/data-dir))
  ([config splits repo data-dir]
   (let [root (paths/data-root repo data-dir)
         missing (remove #(split-complete? config % root) splits)]
     (when (and (seq missing) (nil? (paths/local-repo repo)) (not (hub/offline?)))
       (hub/snapshot-download {:repo repo :repo-type :dataset :local-dir root
                               :patterns (for [s missing] (str "data/" config "/" s "-*.parquet"))}))
     root)))

(defn load-split
  "One split from local parquet files, in stored row order. columns limits
  the columns; names the split lacks are skipped."
  ([config split root] (load-split config split root nil))
  ([config split root columns]
   (let [files (split-files config split root)]
     (when (empty? files)
       (throw (ex-info (str "No parquet files for " config "/" split " under " (fs/path root "data" config) ".")
                       {:type :file-not-found :config config :split split})))
     (when-not (split-complete? config split root)
       (throw (ex-info (str "Some parquet shards of " config "/" split " are missing under "
                            (fs/parent (first files)) ".")
                       {:type :file-not-found :config config :split split})))
     (let [available (pq/column-names (first files))
           columns (if columns (filterv (set available) columns) available)]
       (ds/from-rows (pq/read-rows files columns) columns)))))

(defn memory-placeholder
  "The text appended to the system prompt that reserves the memory positions."
  [num-memories]
  (str "\nMemory" (apply str (repeat num-memories util/memory-token)) "Memory"))

(defn build-messages
  "The chat for one example."
  ([question system-prompt num-memories] (build-messages question system-prompt num-memories "" nil))
  ([question system-prompt num-memories suffix answer]
   (cond-> [{:role "system" :content (str system-prompt (memory-placeholder num-memories))}
            {:role "user" :content (str (str/trim (str question)) suffix)}]
     (some? answer) (conj {:role "assistant" :content (str/trim (str answer))}))))

(defn memory-dataset
  "Concatenation of splits whose items remember their source: parts are
  [config split dataset] triples."
  [parts]
  {:parts (vec parts)
   :offsets (vec (reductions + 0 (map #(ds/row-count (nth % 2)) parts)))})

(defn dataset-size [mds] (peek (:offsets mds)))

(defn dataset-item
  "One row with its config, split and local row index."
  [{:keys [parts offsets]} index]
  (let [part (dec (count (take-while #(<= % index) offsets)))
        [config split dataset] (nth parts part)
        row (- index (nth offsets part))]
    (assoc (ds/row dataset row) "config" config "split" split "row" row)))

(defn- left-pad [ids width pad]
  (into (vec (repeat (- width (count ids)) pad)) ids))

(defn- documents
  "Gather and pad the document embeddings of a batch: [matrices mask]."
  [stores batch]
  (let [rows (mapv (fn [item]
                     (or (:doc-embeds item)
                         (store/documents (get stores [(get item "config") (get item "split")])
                                          (get item collection-column))))
                   batch)
        max-docs (apply max (map :rows rows))
        dim (apply max (map :cols rows))]
    [(mapv (fn [m]
             (let [out (float-array (* max-docs dim))]
               (System/arraycopy (:data m) 0 out 0 (* (:rows m) (:cols m)))
               (t/matrix max-docs dim out)))
           rows)
     (mapv (fn [m] (vec (concat (repeat (:rows m) false) (repeat (- max-docs (:rows m)) true)))) rows)]))

(defn- labels
  "Labels that supervise only the tokens after each prompt."
  [input-ids attention-mask prompt-lengths]
  (mapv (fn [ids mask prompt-length]
          (let [start (+ (count (filter zero? mask)) prompt-length)]
            (vec (map-indexed (fn [i id] (if (or (< i start) (zero? (nth mask i))) ignore-index id)) ids))))
        input-ids attention-mask prompt-lengths))

(defn collate
  "Turn items of a memory dataset into decoder inputs and embeddings.
  Prompts are rendered with the chat template, truncated to max-length and
  left padded. In training mode the answer is appended and :labels
  supervise only the answer tokens; otherwise the prompt ends with the
  assistant header.

  opts: :tokenizer :stores {[config split] store} :num-memories :max-length
  :system-prompt :suffix :train. An item carrying :doc-embeds (a matrix) and
  :question-embed (floats) uses those instead of a store.
  Returns :input-ids and :attention-mask (vectors of equal-length vectors),
  :doc-embeds (one padded matrix per row), :doc-padding-mask, :question-embeds
  (float arrays) and, in training mode, :labels."
  [{:keys [tokenizer stores num-memories max-length system-prompt suffix train]
    :or {system-prompt default-system-prompt suffix "" train true}}
   batch]
  (let [rendered (for [item batch]
                   (let [answer (when train (get item answer-column))
                         messages (build-messages (get item question-column) system-prompt num-memories suffix answer)
                         prompt (tok/apply-chat-template tokenizer (subvec messages 0 2) true)]
                     (if train
                       [(tok/apply-chat-template tokenizer messages false) (count (tok/encode tokenizer prompt))]
                       [prompt nil])))
        encoded (mapv (fn [[text _]] (vec (take max-length (tok/encode tokenizer text)))) rendered)
        width (apply max (map count encoded))
        pad (tok/pad-id tokenizer)
        input-ids (mapv #(left-pad % width pad) encoded)
        attention-mask (mapv #(left-pad (vec (repeat (count %) 1)) width 0) encoded)
        [doc-embeds doc-padding-mask] (documents stores batch)
        question-embeds (mapv (fn [item]
                                (or (:question-embed item)
                                    (store/question (get stores [(get item "config") (get item "split")])
                                                    (get item "row"))))
                              batch)]
    (cond-> {:input-ids input-ids
             :attention-mask attention-mask
             :doc-embeds doc-embeds
             :doc-padding-mask doc-padding-mask
             :question-embeds question-embeds}
      train (assoc :labels (labels input-ids attention-mask (map second rendered))))))
