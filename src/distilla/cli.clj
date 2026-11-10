(ns distilla.cli
  "Command-line parsing shared by the entry points. Flags keep the Python
  scripts' spellings (--max_length, --data_repo, ...)."
  (:require [babashka.cli :as cli]
            [clojure.string :as str]))

(defn- describe [spec]
  (str/join "\n"
            (for [[k {:keys [desc default required]}] spec]
              (format "  --%-28s %s%s%s" (name k) (or desc "")
                      (if (some? default) (str " (default: " default ")") "")
                      (if required " [required]" "")))))

(defn parse
  "Parse args against a spec {flag {:desc :default :coerce :required}};
  prints usage and exits on --help or a missing required flag."
  [usage spec args]
  (let [opts (cli/parse-opts args {:spec spec})]
    (when (:help opts)
      (println usage)
      (println (describe spec))
      (System/exit 0))
    (doseq [[k {:keys [required]}] spec]
      (when (and required (nil? (get opts k)))
        (binding [*out* *err*]
          (println (str "missing --" (name k)))
          (println usage)
          (println (describe spec)))
        (System/exit 2)))
    opts))

(defn exit-on-error
  "Run f, printing an ex-info's message instead of a stack trace."
  [f]
  (try
    (f)
    (catch clojure.lang.ExceptionInfo e
      (binding [*out* *err*]
        (println "error:" (ex-message e)))
      (System/exit 1))))
