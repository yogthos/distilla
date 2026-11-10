(ns distilla.util
  "The memory special token, checkpoint locations, recipes, JSON writers and
  small helpers shared by the other namespaces."
  (:require [babashka.fs :as fs]
            [clojure.data.json :as json]
            [clojure.string :as str]
            [distilla.paths :as paths]
            [jolt.yaml :as yaml]))

(def memory-token "<|memory|>")
(def checkpoint-file-name "memory.pt")
(def safetensors-checkpoint-file-name "memory.safetensors")
(def config-file-name "config.json")
(def wrapper-prefixes ["module." "_orig_mod."])
(def memory-prefix "memory.")
(def default-seed 23)

(defn str->bool
  "Parse a command-line boolean such as true/false/1/0/yes/no."
  [value]
  (let [lowered (str/lower-case (str/trim (str value)))]
    (cond
      (#{"true" "t" "1" "yes" "y"} lowered) true
      (#{"false" "f" "0" "no" "n"} lowered) false
      :else (throw (ex-info (str "Invalid boolean value: " (pr-str value)) {:value value})))))

(defn checkpoint-file
  "The weights file of a checkpoint directory or file path: memory.pt, or
  memory.safetensors when only that one is there."
  [path]
  (let [path (paths/expand-home path)]
    (if (fs/directory? path)
      (let [pt (str (fs/path path checkpoint-file-name))
            st (str (fs/path path safetensors-checkpoint-file-name))]
        (if (and (not (fs/exists? pt)) (fs/exists? st)) st pt))
      (str path))))

(defn read-recipe
  "Read a YAML training recipe into a map with string keys (empty for an empty file)."
  [path]
  (let [text (slurp (str path))]
    (or (when-not (str/blank? text) (yaml/load text)) {})))

;; --- JSON --------------------------------------------------------------------

(defn- indent [n] (apply str (repeat n " ")))

(defn- json-pretty
  "JSON with two-space indentation, laid out like Python's json.dump(indent=2)."
  [value level]
  (cond
    (map? value)
    (if (empty? value)
      "{}"
      (str "{\n"
           (str/join ",\n" (for [[k v] value]
                             (str (indent (* 2 (inc level)))
                                  (json/write-str (if (keyword? k) (name k) (str k)))
                                  ": " (json-pretty v (inc level)))))
           "\n" (indent (* 2 level)) "}"))

    (and (sequential? value) (not (string? value)))
    (if (empty? value)
      "[]"
      (str "[\n"
           (str/join ",\n" (for [v value] (str (indent (* 2 (inc level))) (json-pretty v (inc level)))))
           "\n" (indent (* 2 level)) "]"))

    :else (json/write-str value :escape-unicode false)))

(defn write-json
  "Write a JSON file, creating parent directories."
  [path payload]
  (some-> (fs/parent path) fs/create-dirs)
  (spit (str path) (str (json-pretty payload 0) "\n")))

(defn write-jsonl
  "Write one JSON object per line, creating parent directories."
  [path rows]
  (some-> (fs/parent path) fs/create-dirs)
  (with-open [w (clojure.java.io/writer (str path))]
    (doseq [row rows]
      (.write w (json/write-str row :escape-unicode false))
      (.write w "\n"))))

(defn read-json [path] (json/read-str (slurp (str path))))

(defn read-jsonl [path]
  (->> (str/split-lines (slurp (str path)))
       (remove str/blank?)
       (mapv json/read-str)))

;; --- numbers -----------------------------------------------------------------

(defn round2
  "Round to two decimals, like Python's round(x, 2) for the values reported here."
  [x]
  (/ (Math/round (* 100.0 (double x))) 100.0))

(defn mean [xs]
  (if (seq xs) (/ (reduce + 0.0 xs) (count xs)) Double/NaN))
