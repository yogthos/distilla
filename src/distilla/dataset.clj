(ns distilla.dataset
  "An in-memory table, the counterpart of a datasets.Dataset: an ordered
  list of column names and a vector of rows, each a map from column name
  (string) to value.")

(defn from-rows
  "A dataset from row maps (string keys). Columns follow the first row's key
  order, or are given explicitly."
  ([rows] (from-rows rows (vec (keys (first rows)))))
  ([rows column-names]
   {:column-names (vec column-names) :rows (vec rows)}))

(defn column-names [ds] (:column-names ds))
(defn has-column? [ds name] (boolean (some #{name} (:column-names ds))))
(defn row-count [ds] (count (:rows ds)))
(defn row [ds i] (nth (:rows ds) i))
(defn rows [ds] (:rows ds))

(defn column
  "Every value of one column, in row order."
  [ds name]
  (when-not (has-column? ds name)
    (throw (ex-info (str "no column " (pr-str name)) {:column name :columns (:column-names ds)})))
  (mapv #(get % name) (:rows ds)))

(defn select-columns [ds names]
  (let [names (vec names)]
    {:column-names names :rows (mapv #(select-keys % names) (:rows ds))}))

(defn remove-columns [ds & names]
  (let [drop (set names)
        keep (vec (remove drop (:column-names ds)))]
    (select-columns ds keep)))
