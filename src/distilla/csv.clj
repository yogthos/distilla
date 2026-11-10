(ns distilla.csv
  "CSV as Python's csv module writes and reads it (QUOTE_MINIMAL, \"\\n\" line
  terminator): fields with a comma, quote or line break are quoted, quotes
  doubled."
  (:require [clojure.string :as str]))

(defn- quote-field [v]
  (let [s (if (nil? v) "" (str v))]
    (if (re-find #"[,\"\n\r]" s)
      (str "\"" (str/replace s "\"" "\"\"") "\"")
      s)))

(defn write-rows
  "CSV text: a header of columns, then one line per row map (keys as given)."
  [columns rows]
  (str (str/join "," (map quote-field columns)) "\n"
       (apply str (for [row rows]
                    (str (str/join "," (map #(quote-field (get row %)) columns)) "\n")))))

(defn parse
  "Rows of CSV text as vectors of strings."
  [^String text]
  (let [n (count text)]
    (loop [i 0 field (StringBuilder.) row [] rows [] quoted? false]
      (if (>= i n)
        (let [row (if (or (pos? (.length field)) (seq row)) (conj row (str field)) row)]
          (if (seq row) (conj rows row) rows))
        (let [c (.charAt text i)]
          (cond
            quoted? (cond
                      (and (= c \") (< (inc i) n) (= \" (.charAt text (inc i))))
                      (recur (+ i 2) (.append field \") row rows true)
                      (= c \") (recur (inc i) field row rows false)
                      :else (recur (inc i) (.append field c) row rows true))
            (= c \") (recur (inc i) field row rows true)
            (= c \,) (recur (inc i) (StringBuilder.) (conj row (str field)) rows false)
            (= c \return) (recur (inc i) field row rows false)
            (= c \newline) (recur (inc i) (StringBuilder.) [] (conj rows (conj row (str field))) false)
            :else (recur (inc i) (.append field c) row rows false)))))))

(defn read-maps
  "Rows of CSV text as maps keyed by the header."
  [text]
  (let [[header & rows] (parse text)]
    (mapv #(zipmap header %) rows)))
