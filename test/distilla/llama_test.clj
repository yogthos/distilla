(ns distilla.llama-test
  "The llama.cpp shim, on a GGUF named by DISTILLA_TEST_GGUF (any small model
  works, e.g. ggml-org/models tinyllamas/stories260K.gguf). Skipped when the
  variable is unset or the shim is not built."
  (:require [clojure.test :refer [deftest is testing]]
            [distilla.encoder :as encoder]
            [distilla.llama :as llama]
            [distilla.tensor :as t]
            distilla.tokenizer))

(def gguf (System/getenv "DISTILLA_TEST_GGUF"))

(defn- enabled? [] (and gguf (llama/available?)))

(deftest tokenizer-round-trip
  (when (enabled?)
    (llama/quiet! true)
    (with-open [m (llama/load-model gguf)]
      (let [text "Once upon a time, there was a little girl"
            ids (llama/tokenize m text)]
        (is (seq ids))
        (is (= text (llama/detokenize m ids)))))))

(deftest segments-decode-as-one-sequence
  (when (enabled?)
    (llama/quiet! true)
    (with-open [m (llama/load-model gguf)
                d (llama/decoder m {:n-ctx 256})]
      (let [ids (llama/tokenize m "Once upon a time, there was a little girl")
            whole (llama/generate d [[:tokens ids]] {:max-new-tokens 12})]
        (is (seq whole))
        (testing "greedy decoding is deterministic"
          (is (= whole (llama/generate d [[:tokens ids]] {:max-new-tokens 12}))))
        (testing "the prompt may arrive in several token segments"
          (is (= whole (llama/generate d [[:tokens (subvec ids 0 3)] [:tokens (subvec ids 3)]] {:max-new-tokens 12}))))
        (testing "an embedding segment is accepted between token segments"
          (is (string? (llama/generate d [[:tokens (subvec ids 0 3)]
                                          [:embeddings (t/matrix 2 (:n-embd m))]
                                          [:tokens (subvec ids 3)]]
                                       {:max-new-tokens 4}))))
        (is (thrown? clojure.lang.ExceptionInfo
                     (llama/generate d [[:embeddings (t/matrix 1 (inc (:n-embd m)))]] {:max-new-tokens 1})))))))

(deftest pooled-embeddings
  (when (enabled?)
    (llama/quiet! true)
    (with-open [m (llama/load-model gguf)
                e (llama/encoder m {:n-ctx 256 :n-batch 256 :n-seq-max 2 :pooling :mean})]
      (let [texts ["a girl" "a boy" "a girl" "the end of the story"]
            v (encoder/embed e texts)]
        (is (= [4 (:n-embd m)] [(:rows v) (:cols v)]))
        (is (< (Math/abs (- 1.0 (t/norm (t/row v 1)))) 1e-5))
        (testing "batching does not change a text's vector"
          (is (< (Math/abs (- 1.0 (t/dot (t/row v 0) (t/row v 2)))) 1e-5)))))))

;; --- the training graph ------------------------------------------------------------------------

(defn- cross-entropy [^floats logits target]
  (let [mx (reduce max (seq logits))
        z (reduce + (map #(Math/exp (- % mx)) logits))]
    (- (+ mx (Math/log z)) (aget logits target))))

(defn- memory-rows [n e]
  (t/matrix n e (float-array (map #(* 0.5 (Math/sin (* 0.37 %))) (range (* n e))))))

(deftest training-graph-matches-inference
  (when (enabled?)
    (llama/quiet! true)
    (with-open [m (llama/load-model gguf)
                d (llama/decoder m {:n-ctx 256})
                g (llama/grad-model gguf {:gpu? false})]
      (let [ids (llama/tokenize m "Once upon a time, there was a little girl named Lily")
            e (:n-embd m)
            mem (memory-rows 2 e)
            tokens (vec (concat (subvec ids 0 3) [0 0] (subvec ids 3)))
            mem-index (vec (concat [-1 -1 -1 0 1] (repeat (- (count ids) 3) -1)))
            n (count tokens)]
        (is (= (:n-vocab g) (llama/n-vocab m)))
        (doseq [p [2 5 (- n 2)]]
          (let [target (nth tokens (inc p))
                segments (if (< p 3)
                           [[:tokens (subvec ids 0 (inc p))]]
                           [[:tokens (subvec ids 0 3)] [:embeddings mem] [:tokens (subvec ids 3 (- (inc p) 2))]])
                expected (cross-entropy (llama/last-logits d segments) target)
                {:keys [loss]} (llama/loss-and-grad g {:tokens tokens :mem-index mem-index :memory mem
                                                       :targets (assoc (vec (repeat n -1)) p target)}
                                                    :grad? false)]
            ;; llama.cpp keeps its KV cache in f16; the training graph is f32 throughout
            (is (< (Math/abs (- expected loss)) (* 1e-2 (max 1.0 expected))) (str "position " p))))))))

(deftest memory-gradient-matches-finite-differences
  (when (enabled?)
    (llama/quiet! true)
    (with-open [m (llama/load-model gguf)
                g (llama/grad-model gguf {:gpu? false})]
      (let [ids (llama/tokenize m "Once upon a time, there was a little girl named Lily")
            e (:n-embd m)
            mem (memory-rows 2 e)
            tokens (vec (concat (subvec ids 0 3) [0 0] (subvec ids 3)))
            n (count tokens)
            batch {:tokens tokens :mem-index (vec (concat [-1 -1 -1 0 1] (repeat (- (count ids) 3) -1)))
                   :memory mem :targets (vec (concat (repeat 5 -1) (subvec tokens 6) [-1]))}
            {:keys [grad n-targets]} (llama/loss-and-grad g batch)
            h 1e-2]
        (is (= (- n 6) n-targets))
        (doseq [i [0 7 (+ e 3)]]
          (let [loss-at (fn [delta]
                          (let [d (t/copy-floats (:data mem))]
                            (aset d i (float (+ (aget d i) delta)))
                            (:loss (llama/loss-and-grad g (assoc batch :memory (t/matrix 2 e d)) :grad? false))))
                fd (/ (- (loss-at h) (loss-at (- h))) (* 2 h))
                analytic (aget ^floats (:data grad) i)]
            (is (< (Math/abs (- fd analytic)) (+ 1e-3 (* 1e-2 (Math/abs fd)))) (str "element " i))))))))

(deftest runtime-learning-lowers-the-loss
  (when (enabled?)
    (llama/quiet! true)
    (let [learn! (requiring-resolve 'distilla.train/learn!)
          trainer (requiring-resolve 'distilla.train/trainer)
          llama-decoder (requiring-resolve 'distilla.train/llama-decoder)
          init (requiring-resolve 'distilla.memory/init-module)]
      (with-open [m (llama/load-model gguf)
                  enc (llama/encoder m {:n-ctx 256 :n-batch 256 :pooling :mean})
                  g (llama/grad-model gguf {:gpu? false})
                  module (init {:embedding-dim (:n-embd m) :output-dim (:n-embd m) :num-memories 4
                                :num-heads 4 :dropout 0.0} 1)]
        (let [tr (trainer {:memory module :decoder (llama-decoder g)
                           :tokenizer (llama/tokenizer m distilla.tokenizer/chatml) :learning-rate 3e-3})
              results (learn! tr enc [{:question "What was the dog called?" :answer "The dog was called Spot."
                                       :documents ["Tom had a big dog." "The dog was named Spot."]}]
                              {:steps 25 :batch-size 1})]
          (is (< (:loss (peek results)) (- (:loss (first results)) 0.5))))))))

;; --- the KV cache as a continuation store ------------------------------------------------------

(deftest reused-prefix-matches-a-fresh-decode
  (when (enabled?)
    (llama/quiet! true)
    (with-open [m (llama/load-model gguf)
                d (llama/decoder m {:n-ctx 512})
                fresh (llama/decoder m {:n-ctx 512})]
      (let [prefix (llama/tokenize m "Once upon a time, there was a little girl")
            mem (memory-rows 3 (:n-embd m))
            ask (fn [tail] [[:tokens prefix] [:embeddings mem] [:tokens (llama/tokenize m tail)]])
            ;; llama.cpp's numerics depend a little on batch shape (and its KV cache is f16), so a
            ;; reused cache agrees with a fresh decode to ~1e-3 relative, not bit for bit
            same-logits? (fn [segments]
                           (let [a (llama/last-logits d segments)
                                 b (llama/last-logits fresh segments {:reuse-prefix false})
                                 scale (reduce max 1.0 (map #(Math/abs (double %)) b))]
                             (< (reduce max (map #(Math/abs (- %1 %2)) a b)) (* 5e-3 scale))))]
        (testing "prompts sharing a prefix (and memory rows)"
          (is (same-logits? (ask " named Lily.")))
          (is (same-logits? (ask " who liked cats."))))
        (testing "a prompt that diverges inside the memory rows"
          (is (same-logits? [[:tokens prefix] [:embeddings (memory-rows 2 (:n-embd m))]]))
          (is (same-logits? [[:tokens prefix] [:embeddings (t/matrix 3 (:n-embd m))] [:tokens [5]]])))
        (testing "a continuation of the previous prompt and its generated reply"
          (let [first-turn [[:tokens prefix]]
                {:keys [tokens text]} (llama/generate d first-turn {:max-new-tokens 8 :tokens? true})
                second-turn [[:tokens (into (into prefix tokens) (llama/tokenize m " Then"))]]]
            (is (= text (llama/generate fresh first-turn {:max-new-tokens 8 :reuse-prefix false})))
            (is (same-logits? second-turn))))
        (testing "the same prompt twice"
          (is (= (llama/generate d (ask " named Lily.") {:max-new-tokens 6})
                 (llama/generate d (ask " named Lily.") {:max-new-tokens 6}))))))))

(deftest layerwise-backward-matches-the-single-graph
  (when (enabled?)
    (llama/quiet! true)
    (with-open [m (llama/load-model gguf)
                g (llama/grad-model gguf {:gpu? false})]
      (let [ids (llama/tokenize m "Once upon a time, there was a little girl named Lily who liked to play")
            e (:n-embd m)
            mem (memory-rows 3 e)
            tokens (vec (concat (subvec ids 0 3) [0 0 0] (subvec ids 3)))
            n (count tokens)
            batch {:tokens tokens :mem-index (vec (concat [-1 -1 -1 0 1 2] (repeat (- (count ids) 3) -1)))
                   :memory mem :targets (vec (concat (repeat 6 -1) (subvec tokens 7) [-1]))}
            _ (llama/set-chunking! g {:full-max 100000})
            full (llama/loss-and-grad g batch)
            _ (llama/set-chunking! g {:full-max 0 :chunk 5})
            layerwise (llama/loss-and-grad g batch)
            loss-only (llama/loss-and-grad g batch :grad? false)]
        (llama/set-chunking! g {:full-max 1024 :chunk 256})
        (is (= (:n-targets full) (:n-targets layerwise)))
        (is (< (Math/abs (- (:loss full) (:loss layerwise))) 1e-4))
        (is (< (Math/abs (- (:loss full) (:loss loss-only))) 1e-4))
        (is (< (reduce max (map #(Math/abs (- %1 %2)) (:data (:grad full)) (:data (:grad layerwise))))
               (* 1e-3 (reduce max 1e-6 (map #(Math/abs (double %)) (:data (:grad full)))))))))))
