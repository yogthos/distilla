(ns distilla.memory-test
  "The memory module: parity with the PyTorch forward pass (fixtures made by
  dev/make_fixtures.py), shapes, masking, architecture options and
  checkpoint I/O."
  (:require [babashka.fs :as fs]
            [clojure.test :refer [deftest is testing]]
            [distilla.helpers :as h]
            [distilla.memory :as memory]
            [distilla.tensor :as t]
            [distilla.util :as util]))

(def fixtures "test/fixtures/memory")
(def expected (delay (util/read-json (str fixtures "/expected.json"))))

(defn- inputs []
  (let [{:strs [docs mask question]} (get @expected "inputs")]
    [(mapv t/from-rows docs) mask (mapv #(float-array (map float %)) question)]))

(defn- run [m docs mask questions]
  (mapv t/rows->vecs (memory/forward-batch m docs mask questions)))

(deftest matches-pytorch
  (let [[docs mask questions] (inputs)]
    (doseq [[name v] (get @expected "variants")]
      (testing name
        (with-open [m (memory/from-pretrained (str fixtures "/" name))]
          ;; dropout leaves no trace in the weights: a module without config.json gets the default
          (is (= (dissoc (memory/json->config (get v "config")) :dropout) (dissoc (:config m) :dropout)))
          (is (h/approx= (run m docs mask questions) (get v "masked") 1e-4))
          (is (h/approx= (run m docs nil questions) (get v "unmasked") 1e-4))
          (is (h/approx= (run m [(t/take-rows (second docs) 3)] nil [(second questions)]) (get v "truncated") 1e-4))
          (is (h/approx= (run m docs mask (mapv #(float-array (map - %)) questions)) (get v "negated") 1e-4)))))))

(defn- config [& {:as overrides}]
  (memory/config (merge {:embedding-dim 32 :output-dim 48 :num-memories 4 :num-heads 4 :dropout 0.0} overrides)))

(defn- make-memory [& overrides]
  (let [cfg (apply config overrides)]
    (memory/from-state cfg (h/random-state cfg 11))))

(defn- random-inputs []
  (let [docs [(t/matrix 6 32 (h/normals 1 (* 6 32))) (t/matrix 6 32 (h/normals 2 (* 6 32)))]
        mask [[false false false false false false] [false false false true true true]]
        questions [(h/normals 3 32) (h/normals 4 32)]]
    [docs mask questions]))

(deftest output-shape
  (let [[docs mask questions] (random-inputs)]
    (doseq [layers [0 1 2] retrieval [false true]]
      (with-open [m (make-memory :num-self-attn-layers layers :retrieval-init retrieval)]
        (doseq [out (concat (memory/forward-batch m docs mask questions)
                            (memory/forward-batch m docs nil questions))]
          (is (= [4 48] [(:rows out) (:cols out)])))))))

(deftest padding-positions-are-ignored
  (let [[docs mask questions] (random-inputs)]
    (doseq [layers [0 1]]
      (with-open [m (make-memory :num-self-attn-layers layers)]
        (let [reference (run m docs mask questions)
              perturbed (update docs 1 (fn [d] (let [data (t/copy-floats (:data d))]
                                                 (doseq [i (range (* 3 32) (* 6 32))] (aset data i (float 100.0)))
                                                 (assoc d :data data))))]
          (is (h/approx= (run m perturbed mask questions) reference 1e-6))
          (is (h/approx= (run m [(t/take-rows (second docs) 3)] nil [(second questions)]) [(second reference)] 1e-5)))))))

(deftest retrieval-init-uses-most-similar-documents
  (with-open [m (make-memory :retrieval-init true)]
    (is (not (contains? (:params m) "memory_queries")))
    (let [e 32
          question (doto (float-array e) (aset 0 (float 1.0)))
          docs (t/matrix 3 e (let [d (float-array (* 3 e))]
                               (aset d 0 (float 0.1)) (aset d 1 (float 1.0))
                               (aset d e (float 5.0))
                               (aset d (* 2 e) (float 1.0)) (aset d (inc (* 2 e)) (float 1.0))
                               d))
          slots (t/matrix 4 e (memory/initial-slots m docs [false false true] question))]
      (is (= (vec (t/row slots 0)) (vec (t/row docs 1))))
      (is (= (vec (t/row slots 1)) (vec (t/row docs 0))))
      (is (every? zero? (t/row slots 3)))
      (testing "without padding every document is a slot, most similar first, and the rest is zero"
        (let [slots (t/matrix 4 e (memory/initial-slots m docs question))]
          (is (= (mapv #(vec (t/row docs %)) [1 2 0]) (mapv #(vec (t/row slots %)) [0 1 2])))
          (is (every? zero? (t/row slots 3))))))))

(deftest question-conditions-the-output
  (let [[docs mask questions] (random-inputs)]
    (with-open [m (make-memory)]
      (is (not (h/approx= (run m docs mask questions)
                          (run m docs mask (mapv #(float-array (map - %)) questions))
                          1e-6))))))

(deftest save-and-from-pretrained-round-trip
  (h/with-temp-dir [dir]
    (with-open [m (make-memory :num-self-attn-layers 1 :retrieval-init true :num-heads 2)]
      (memory/save m dir)
      (is (fs/exists? (fs/path dir util/safetensors-checkpoint-file-name)))
      (is (= 2 (get (util/read-json (fs/path dir util/config-file-name)) "num_heads")))
      (let [[docs mask questions] (random-inputs)
            expected (run m docs mask questions)]
        (doseq [path [dir (str (fs/path dir util/safetensors-checkpoint-file-name))]]
          (with-open [restored (memory/from-pretrained path)]
            (is (= (:config m) (:config restored)))
            (is (h/approx= (run restored docs mask questions) expected 1e-6))))))))

(deftest config-inferred-from-state-dict
  (doseq [options [{}
                   {:num-self-attn-layers 2 :num-cross-attn-layers 1}
                   {:retrieval-init true :num-self-attn-layers 1 :num-memories memory/default-num-memories}
                   {:num-memories 7 :num-cross-attn-layers 3}]]
    (let [cfg (memory/config (merge {:embedding-dim 32 :output-dim 48 :num-memories 4 :num-heads 8 :dropout 0.1}
                                    options))]
      (is (= cfg (memory/config-from-state-dict (h/random-state cfg 5))) (pr-str options))))
  (testing "a torch checkpoint without config.json"
    (with-open [m (memory/from-pretrained (str fixtures "/inferred"))]
      (is (= 7 (memory/num-memories m)))
      (is (= 3 (get-in m [:config :num-cross-attn-layers]))))))

(deftest load-memory-state-unwraps-checkpoints
  (let [plain (memory/load-memory-state (str fixtures "/default/memory.pt"))]
    (doseq [file ["module.pt" "prefixed.pt" "lightning.pt"]]
      (let [loaded (memory/load-memory-state (str fixtures "/wrapped/" file))]
        (is (= (set (keys plain)) (set (keys loaded))) file)
        (doseq [[k v] plain]
          (is (= (vec (:data v)) (vec (:data (get loaded k)))) (str file " " k)))))))

(deftest strict-loading
  (let [cfg (config)
        state (h/random-state cfg 3)]
    (is (thrown? clojure.lang.ExceptionInfo (memory/from-state cfg (dissoc state "doc_norm.weight"))))
    (is (thrown? clojure.lang.ExceptionInfo (memory/from-state cfg (assoc state "extra" (get state "doc_norm.bias")))))))

(deftest checkpoints-under-env
  (when-let [root (System/getenv "DISTILLA_TEST_CHECKPOINTS")]
    (doseq [path (fs/glob root "**memory.pt")]
      (with-open [m (memory/from-pretrained (str path))]
        (let [{:keys [embedding-dim output-dim num-memories]} (:config m)
              out (memory/forward m (t/matrix 5 embedding-dim (h/normals 9 (* 5 embedding-dim)))
                                  (h/normals 10 embedding-dim))]
          (is (= [num-memories output-dim] [(:rows out) (:cols out)]))
          (is (every? #(Float/isFinite %) (:data out))))))))
