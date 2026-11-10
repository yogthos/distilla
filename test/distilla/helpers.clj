(ns distilla.helpers
  "Test helpers: a word-level chat tokenizer, a synthetic data repository
  writer, a deterministic fake encoder and random memory-module weights."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [distilla.dataset :as ds]
            [distilla.encoder :as encoder]
            [distilla.parquet :as pq]
            [distilla.paths :as paths]
            [distilla.store :as store]
            [distilla.tensor :as t]
            [distilla.tokenizer :as tok]
            [distilla.util :as util]))

(def embedding-dim 8)

;; --- deterministic pseudo-random numbers -----------------------------------------

(defn- splitmix [state]
  (let [z (unchecked-add (long state) (unchecked-long -7046029254386353131))
        z (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 30)) (unchecked-long -4658895280553007687))
        z (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 27)) (unchecked-long -7723592293110705685))]
    [(bit-xor z (unsigned-bit-shift-right z 31)) (unchecked-add (long state) (unchecked-long -7046029254386353131))]))

(defn normals
  "n standard normal floats from a seed (splitmix64 + Box-Muller)."
  [seed n]
  (loop [state seed out []]
    (if (>= (count out) n)
      (float-array (take n out))
      (let [[a state] (splitmix state)
            [b state] (splitmix state)
            u1 (max 1e-12 (/ (double (unsigned-bit-shift-right a 11)) 9007199254740992.0))
            u2 (/ (double (unsigned-bit-shift-right b 11)) 9007199254740992.0)
            r (Math/sqrt (* -2.0 (Math/log u1)))]
        (recur state (conj out (* r (Math/cos (* 2 Math/PI u2))) (* r (Math/sin (* 2 Math/PI u2)))))))))

(defn crc32 [^String s]
  (let [c (java.util.zip.CRC32.)]
    (.update c (.getBytes s "UTF-8"))
    (.getValue c)))

;; --- fake encoder -------------------------------------------------------------------

(defrecord FakeEncoder [dim table calls]
  encoder/Encoder
  (embed [_ texts]
    (swap! calls conj (vec texts))
    (t/from-rows (for [text texts]
                   (if-let [v (get table text)] (float-array (map float v)) (normals (crc32 text) dim)))
                 dim)))

(defn fake-encoder
  "A deterministic stand-in for an embedding engine: each text maps to a fixed
  pseudo-random vector (or the vector table gives it). Records its calls."
  ([] (fake-encoder embedding-dim {}))
  ([dim table] (->FakeEncoder dim table (atom []))))

(defn vector-of
  "The vector the fake encoder gives text, rounded to float16 as stored."
  [enc text]
  (t/round-to-half (t/row (encoder/embed (assoc enc :calls (atom [])) [text]) 0)))

;; --- word-level tokenizer ----------------------------------------------------------------

(def special-tokens ["[UNK]" "<|endoftext|>" "<|im_start|>" "<|im_end|>"])
(def words (str/split (str "system user assistant Memory You are a helpful what is the capital of france paris "
                           "who wrote hamlet shakespeare answer briefly please suffix ok")
                      #" "))

(defrecord WordTokenizer [vocab inverse]
  tok/Tokenizer
  (encode-text [_ text]
    (let [parts (str/split text #"(?=<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>)|(?<=<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>)")]
      (vec (mapcat (fn [part]
                     (if (contains? #{"<|im_start|>" "<|im_end|>" "<|endoftext|>"} part)
                       [(vocab part)]
                       (map #(get vocab % (vocab "[UNK]")) (re-seq #"\w+|[^\w\s]+" part))))
                   parts))))
  (decode-ids [_ ids] (str/join " " (map inverse ids)))
  (pad-id [_] (vocab "<|endoftext|>"))
  (memory-token-id [_] (vocab util/memory-token))
  (chat-template [_] tok/chatml))

(defn word-tokenizer
  "A word-level tokenizer with a ChatML template; <|memory|> is the last id,
  as when the Python tokenizer registers it."
  []
  (let [tokens (conj (into special-tokens words) util/memory-token)
        vocab (zipmap tokens (range))]
    (->WordTokenizer vocab (zipmap (range) tokens))))

;; --- synthetic data repository ---------------------------------------------------------------

(defn make-rows
  "Two collections, one shared by two rows."
  []
  [{"collection_id" 7 "question" " what is the capital of france " "answer" "paris"
    "documents" ["paris is the capital" "france"] "choices" ["paris" "" "hamlet"] "hard" true}
   {"collection_id" 3 "question" "who wrote hamlet" "answer" "shakespeare"
    "documents" ["hamlet" "shakespeare wrote hamlet" "who" "ok"] "choices" ["paris" "shakespeare" ""] "hard" false}
   {"collection_id" 7 "question" "what is the capital of france" "answer" "paris"
    "documents" ["paris is the capital" "france"] "choices" ["hamlet" "paris" ""] "hard" false}])

(def columns ["collection_id" "question" "answer" "documents" "choices" "hard"])

(defn write-split
  "Write a split's parquet file and, when enc is given, its embeddings. Returns the dataset."
  ([root config split rows] (write-split root config split rows nil))
  ([root config split rows enc]
   (let [cols (filterv #(contains? (first rows) %) columns)]
     (pq/write-rows (str (fs/path root "data" config (str split "-00000-of-00001.parquet"))) rows cols)
     (when enc
       (let [first-docs (reduce (fn [m row] (if (contains? m (get row "collection_id"))
                                              m
                                              (assoc m (get row "collection_id") (get row "documents"))))
                                {} rows)]
         (store/with-writer [w (store/embedding-dir config split root paths/default-encoder)]
           (doseq [id (sort (keys first-docs))]
             (store/add! w id (t/from-rows (map #(vector-of enc %) (get first-docs id)) (:dim enc))))
           (store/write-questions! w (t/from-rows (map #(vector-of enc (str/trim (get % "question"))) rows)
                                                  (:dim enc))))))
     (ds/from-rows rows cols))))

(defn temp-dir []
  (str (fs/create-temp-dir {:prefix "distilla-test-"})))

(defmacro with-temp-dir [[sym] & body]
  `(let [~sym (temp-dir)]
     (try ~@body (finally (fs/delete-tree ~sym)))))

(defn data-repo
  "A local data repository with one config (toy) holding train and test splits with embeddings."
  [root enc]
  (let [rows (make-rows)]
    (write-split root "toy" "train" rows enc)
    (write-split root "toy" "test" (subvec rows 0 2) enc)
    root))

;; --- random memory weights ----------------------------------------------------------------

(defn random-state
  "A state dict for a memory configuration, filled with small random values."
  [{:keys [embedding-dim output-dim num-memories num-self-attn-layers num-cross-attn-layers retrieval-init]} seed]
  (let [e embedding-dim
        counter (atom seed)
        rand-t (fn [shape scale]
                 (let [n (reduce * shape)
                       data (normals (swap! counter inc) n)]
                   (dotimes [i n] (aset data i (float (* scale (aget data i)))))
                   {:dtype "F32" :shape shape :data data}))
        ones (fn [n] {:dtype "F32" :shape [n] :data (float-array (repeat n 1.0))})
        attention (fn [p] {(str p "in_proj_weight") (rand-t [(* 3 e) e] 0.3)
                           (str p "in_proj_bias") (rand-t [(* 3 e)] 0.1)
                           (str p "out_proj.weight") (rand-t [e e] 0.3)
                           (str p "out_proj.bias") (rand-t [e] 0.1)})
        norm (fn [p] {(str p "weight") (ones e) (str p "bias") (rand-t [e] 0.1)})
        linear (fn [p out in] {(str p "weight") (rand-t [out in] 0.3) (str p "bias") (rand-t [out] 0.1)})]
    (merge
     (when-not retrieval-init {"memory_queries" (rand-t [1 num-memories e] 1.0)})
     (attention "query_cond_attn.")
     (norm "query_cond_norm.")
     (apply merge {} (for [i (range num-self-attn-layers)]
                       (let [p (str "encoder.layers." i ".")]
                         (merge (attention (str p "self_attn."))
                                (linear (str p "linear1.") (* 4 e) e)
                                (linear (str p "linear2.") e (* 4 e))
                                (norm (str p "norm1.")) (norm (str p "norm2."))))))
     (norm "doc_norm.")
     (apply merge {} (for [i (range num-cross-attn-layers)]
                       (merge (norm (str "cross_attn_norms." i "."))
                              (attention (str "cross_attn_layers." i "."))
                              (norm (str "ffn_norms." i "."))
                              (linear (str "ffn_layers." i ".0.") (* 4 e) e)
                              (linear (str "ffn_layers." i ".3.") e (* 4 e)))))
     (norm "layer_norm.")
     (linear "memory_projection." output-dim e))))

(defn approx=
  "Element-wise closeness of two nested number sequences."
  ([a b] (approx= a b 1e-5))
  ([a b tol]
   (let [fa (flatten (if (instance? (class (float-array 0)) a) (vec a) a))
         fb (flatten (if (instance? (class (float-array 0)) b) (vec b) b))]
     (and (= (count fa) (count fb))
          (every? true? (map (fn [x y] (<= (Math/abs (- (double x) (double y))) (+ tol (* tol (Math/abs (double y))))))
                             fa fb))))))
