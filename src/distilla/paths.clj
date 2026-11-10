(ns distilla.paths
  "Default locations and Hub repository ids.

  The data, results and model-cache locations and the data repository can be
  set with the DISTILLA_* environment variables; every entry point also takes
  explicit options."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]))

(defn- env [k default] (or (not-empty (System/getenv k)) default))

(defn expand-home [path]
  (let [path (str path)]
    (if (str/starts-with? path "~")
      (str (System/getProperty "user.home") (subs path 1))
      path)))

(def home (expand-home (env "DISTILLA_HOME" ".")))
(def data-dir (expand-home (env "DISTILLA_DATA_DIR" (str (fs/path home "data")))))
(def results-dir (str (fs/path home "results")))
(def model-cache-dir (some-> (System/getenv "DISTILLA_MODEL_CACHE") not-empty expand-home))

(def data-repo (env "DISTILLA_DATA_REPO" "memorilla/Memorilla-Data"))

;; Hub ids of the models the released checkpoints were trained with. They name
;; the embedding folders of the data repository (qwen3-embedding-4b) and the
;; checkpoints; the models themselves run from GGUF files (--decoder,
;; --encoder).
(def default-decoder "Qwen/Qwen3-8B")
(def default-encoder "Qwen/Qwen3-Embedding-4B")

(defn resolve-path
  "Resolve a path against DISTILLA_HOME unless it is already absolute."
  ([path] (resolve-path home path))
  ([base path]
   (let [path (expand-home path)]
     (if (fs/absolute? path) path (str (fs/path base path))))))

(defn local-repo
  "The local directory a data repository refers to, or nil for a Hub repo id."
  [repo]
  (let [path (resolve-path repo)]
    (when (fs/directory? path) path)))

(defn data-root
  "The local directory holding a data repository's data/ and embeddings/."
  ([] (data-root data-repo data-dir))
  ([repo dir] (or (local-repo repo) (resolve-path dir))))

(defn encoder-slug
  "The directory name used for an encoder's embeddings, e.g. qwen3-embedding-4b."
  [encoder]
  (str/lower-case (last (str/split (str/replace encoder #"/+$" "") #"/"))))
