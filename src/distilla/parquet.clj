(ns distilla.parquet
  "Parquet reading and writing through DuckDB's C API (native/libduckdb,
  fetched by `jolt native`).

  Rows travel as JSON: DuckDB renders each row with to_json, so list columns
  (documents, choices) arrive as vectors, booleans as booleans and integers
  as longs, without binding DuckDB's vector API."
  (:require [babashka.fs :as fs]
            [clojure.data.json :as json]
            [clojure.string :as str]
            [distilla.util :as util]
            [jolt.ffi :as ffi]))

(ffi/defcfn duckdb-open "duckdb_open" [:string :pointer] :int)
(ffi/defcfn duckdb-close "duckdb_close" [:pointer] :void)
(ffi/defcfn duckdb-connect "duckdb_connect" [:pointer :pointer] :int)
(ffi/defcfn duckdb-disconnect "duckdb_disconnect" [:pointer] :void)
(ffi/defcfn duckdb-query "duckdb_query" [:pointer :pointer :pointer] :int :blocking)
(ffi/defcfn duckdb-destroy-result "duckdb_destroy_result" [:pointer] :void)
(ffi/defcfn duckdb-result-error "duckdb_result_error" [:pointer] :pointer)
(ffi/defcfn duckdb-row-count "duckdb_row_count" [:pointer] :uint64)
(ffi/defcfn duckdb-column-count "duckdb_column_count" [:pointer] :uint64)
(ffi/defcfn duckdb-value-varchar "duckdb_value_varchar" [:pointer :uint64 :uint64] :pointer)
(ffi/defcfn duckdb-free "duckdb_free" [:pointer] :void)

(def ^:private result-size 48) ; sizeof(duckdb_result): six pointer-sized fields

(defn available? []
  (some? (ffi/find-symbol "duckdb_open")))

(defn- check-available []
  (when-not (available?)
    (throw (ex-info "DuckDB is not loaded: run `jolt native` to fetch native/libduckdb" {}))))

(defonce ^:private conn (atom nil))

(defn- connection
  "The process-wide in-memory DuckDB connection, opened on first use."
  []
  (or @conn
      (locking conn
        (or @conn
            (do (check-available)
                (let [dbp (ffi/alloc 8) cp (ffi/alloc 8)]
                  (when-not (zero? (duckdb-open ":memory:" dbp))
                    (throw (ex-info "duckdb_open failed" {})))
                  (when-not (zero? (duckdb-connect (ffi/read dbp :pointer) cp))
                    (throw (ex-info "duckdb_connect failed" {})))
                  (reset! conn {:db dbp :con (ffi/read cp :pointer)})))))))

(defn query
  "Run SQL; the result as a vector of rows, each a vector of strings (nil for NULL)."
  [sql]
  (let [{:keys [con]} (connection)]
    (locking conn
      (ffi/with-alloc [res result-size]
        (let [rc (ffi/with-c-string [q sql] (duckdb-query con q res))]
          (try
            (when-not (zero? rc)
              (let [err (duckdb-result-error res)]
                (throw (ex-info (str "DuckDB: " (if (ffi/null? err) "query failed" (ffi/ptr->string err)))
                                {:sql sql}))))
            (let [rows (duckdb-row-count res)
                  cols (duckdb-column-count res)]
              (vec (for [r (range rows)]
                     (vec (for [c (range cols)]
                            (let [p (duckdb-value-varchar res c r)]
                              (when-not (ffi/null? p)
                                (try (ffi/ptr->string p) (finally (duckdb-free p))))))))))
            (finally (duckdb-destroy-result res))))))))

(defn sql-string [s] (str "'" (str/replace (str s) "'" "''") "'"))
(defn sql-ident [s] (str "\"" (str/replace (str s) "\"" "\"\"") "\""))

(defn- file-list [files] (str "[" (str/join ", " (map sql-string files)) "]"))

(defn column-names
  "The column names of a parquet file, in order."
  [file]
  (mapv first (query (str "SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet("
                          (sql-string file) "))"))))

(defn row-count [files]
  (parse-long (ffirst (query (str "SELECT count(*) FROM read_parquet(" (file-list files) ")")))))

(defn read-rows
  "Rows of parquet files, in file order then stored order, as maps from
  column name to value. columns limits the columns (default every column)."
  ([files] (read-rows files nil))
  ([files columns]
   (let [files (mapv str files)
         columns (or columns (column-names (first files)))]
     (if (empty? columns)
       []
       (let [fields (str/join ", " (for [c columns] (str (sql-ident c) " := " (sql-ident c))))
             sql (str "SELECT to_json(struct_pack(" fields "))::VARCHAR FROM read_parquet(" (file-list files)
                      ", filename = true, file_row_number = true) ORDER BY filename, file_row_number")]
         (mapv (fn [[text]] (json/read-str text)) (query sql)))))))

(defn write-rows
  "Write row maps to a parquet file with the given columns, in that order.
  Column types are inferred by DuckDB from the values (strings, numbers,
  booleans, lists of strings)."
  [path rows columns]
  (let [tmp (str (fs/create-temp-file {:prefix "distilla-" :suffix ".jsonl"}))]
    (try
      (util/write-jsonl tmp (map #(into (array-map) (for [c columns] [c (get % c)])) rows))
      (some-> (fs/parent path) fs/create-dirs)
      (query (str "COPY (SELECT " (str/join ", " (map sql-ident columns))
                  " FROM read_json(" (sql-string tmp) ", format = 'newline_delimited'))"
                  " TO " (sql-string path) " (FORMAT parquet)"))
      (str path)
      (finally (fs/delete-if-exists tmp)))))
