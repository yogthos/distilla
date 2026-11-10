(ns distilla.encoder
  "The text-embedding seam. An encoder turns texts into one pooled,
  L2-normalised vector each; the llama.cpp engine (distilla.llama) and the
  tests' fake encoder both implement it."
  (:require [distilla.tensor :as t]))

(defprotocol Encoder
  (embed [encoder texts] "Embed texts: a distilla.tensor matrix [count(texts) x D] of normalised vectors."))

(def embed-chunk-size 200000)

(defn embed-texts
  "Embed texts in chunks, the result rounded to float16 like the stored
  embeddings: a matrix [n x D]. No texts gives a 0 x 0 matrix."
  ([encoder texts] (embed-texts encoder texts embed-chunk-size))
  ([encoder texts chunk-size]
   (let [texts (vec texts)]
     (if (empty? texts)
       (t/matrix 0 0)
       (let [parts (mapv #(embed encoder %) (partition-all chunk-size texts))
             cols (:cols (first parts))
             m (t/concat-rows parts cols)]
         (assoc m :data (t/round-to-half (:data m))))))))
