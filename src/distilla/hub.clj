(ns distilla.hub
  "Minimal Hugging Face Hub downloads: list a repository's files, keep those
  matching glob patterns, fetch the missing ones with curl. HF_TOKEN is sent
  when set; HF_HUB_OFFLINE=1 turns every download into a no-op."
  (:require [babashka.fs :as fs]
            [clojure.data.json :as json]
            [clojure.string :as str]
            [jolt.process :as p]))

(def endpoint (or (not-empty (System/getenv "HF_ENDPOINT")) "https://huggingface.co"))

(defn offline? []
  (contains? #{"1" "true" "yes" "on"} (some-> (System/getenv "HF_HUB_OFFLINE") str/lower-case)))

(defn- auth-args []
  (when-let [token (not-empty (System/getenv "HF_TOKEN"))]
    ["-H" (str "Authorization: Bearer " token)]))

(defn- curl [& args]
  (let [{:keys [exit out err]} (apply p/sh (concat ["curl" "-sSfL"] (auth-args) args))]
    (when-not (zero? exit)
      (throw (ex-info (str "curl failed: " (str/trim (str err))) {:args args :exit exit})))
    out))

(defn glob->regex
  "A glob (* within a path segment, ** across segments, ?) as a regex."
  [glob]
  (re-pattern
   (str "^"
        (-> (java.util.regex.Pattern/quote glob)
            (str/replace "**" "\u0000")
            (str/replace "*" "\\E[^/]*\\Q")
            (str/replace "\u0000" "\\E.*\\Q")
            (str/replace "?" "\\E[^/]\\Q"))
        "$")))

(defn- repo-path [repo-type repo]
  (case repo-type :dataset (str "datasets/" repo) :model (str "models/" repo)))

(defn list-files
  "Every file path of a repository under prefix (recursively)."
  [repo repo-type prefix]
  (let [url (str endpoint "/api/" (repo-path repo-type repo) "/tree/main"
                 (when (seq prefix) (str "/" prefix)) "?recursive=true")
        entries (json/read-str (curl url))]
    (vec (for [e entries :when (= "file" (get e "type"))] (get e "path")))))

(defn- literal-prefix
  "The directory part of a glob before its first wildcard."
  [glob]
  (let [idx (or (first (keep-indexed (fn [i c] (when (#{\* \?} c) i)) glob)) (count glob))
        head (subs glob 0 idx)]
    (if-let [slash (str/last-index-of head "/")] (subs head 0 slash) "")))

(defn snapshot-download
  "Download the files of repo matching any of patterns into local-dir,
  skipping files already present. Returns local-dir."
  [{:keys [repo repo-type patterns local-dir] :or {repo-type :model}}]
  (when-not (offline?)
    (let [prefixes (distinct (map literal-prefix patterns))
          regexes (map glob->regex patterns)
          files (distinct (mapcat #(list-files repo repo-type %) prefixes))
          wanted (filter (fn [f] (some #(re-matches % f) regexes)) files)]
      (doseq [f wanted]
        (let [target (fs/path local-dir f)]
          (when-not (fs/exists? target)
            (fs/create-dirs (fs/parent target))
            (let [tmp (str target ".incomplete")
                  url (str endpoint "/" (if (= repo-type :dataset) (str "datasets/" repo) repo)
                           "/resolve/main/" f)]
              (println "downloading" f)
              (curl "-o" tmp url)
              (fs/move tmp target {:replace-existing true})))))))
  (str local-dir))
