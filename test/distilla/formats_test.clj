(ns distilla.formats-test
  "The torch checkpoint reader, safetensors and float16 conversion."
  (:require [babashka.fs :as fs]
            [clojure.test :refer [deftest is testing]]
            [distilla.helpers :as h]
            [distilla.safetensors :as st]
            [distilla.tensor :as t]
            [distilla.torch :as torch]
            [distilla.util :as util]))

(deftest unpickles-torch-checkpoints
  (let [m (torch/load "test/fixtures/memory/wrapped/misc.pt")]
    (testing "non-contiguous tensors are gathered through their strides"
      (is (= [3 2] (:shape (m "transposed"))))
      (is (= [0.0 3.0 1.0 4.0 2.0 5.0] (map double (get-in m ["transposed" :data]))))
      (is (= [2.0 3.0 4.0 5.0 6.0] (map double (get-in m ["sliced" :data])))))
    (is (= [0.5 -1.25 3.0] (map double (get-in m ["half" :data]))))
    (is (= [1 -2 3000000000] (vec (get-in m ["long" :data]))))
    (is (= [true 7 0.25 "x"] [(m "flag") (m "count") (m "ratio") (m "name")]))
    (is (= {"list" [1 2 3] "tuple" [1 "a"]} (m "nested")))))

(deftest reads-python-safetensors
  (let [s (st/open "test/fixtures/safetensors/sample.safetensors")
        expected (util/read-json "test/fixtures/safetensors/expected.json")]
    (doseq [[name {:strs [shape data]}] expected]
      (let [tensor (st/read-tensor s name)]
        (is (= shape (:shape tensor)) name)
        (is (h/approx= (vec (:data tensor)) data 1e-6) name)))
    (is (h/approx= [1.0E-4 0.0 -0.0 7.0 1.5 2.5 -3.5 4.5] (vec (:data (st/read-rows s "f16" 1 3))) 1e-3))))

(deftest safetensors-round-trip
  (h/with-temp-dir [dir]
    (let [path (str (fs/path dir "x.safetensors"))
          data (h/normals 1 12)]
      (st/save-file {"a" {:dtype "F32" :shape [3 4] :data data}
                     "h" {:dtype "F16" :shape [12] :data data}
                     "b" {:dtype "BF16" :shape [2 6] :data data}
                     "i" {:dtype "I64" :shape [3] :data [1 -2 (bit-shift-left 1 40)]}
                     "j" {:dtype "I32" :shape [2] :data [5 -6]}}
                    path {"format" "pt"})
      (let [s (st/open path)]
        (is (= {"format" "pt"} (:metadata s)))
        (is (= (vec data) (vec (:data (st/read-tensor s "a")))))
        (is (= (vec (t/round-to-half data)) (vec (:data (st/read-tensor s "h")))))
        (is (h/approx= (vec (:data (st/read-tensor s "b"))) (vec data) 1e-2))
        (is (= [1 -2 (bit-shift-left 1 40)] (vec (:data (st/read-tensor s "i")))))
        (is (= [5 -6] (vec (:data (st/read-tensor s "j")))))
        (is (zero? (mod (:data-start s) 8)))))))

(deftest float16-conversion
  (is (= [15360 49408 31743 31744 2 11878] (map t/float->half [1.0 -2.5 65504.0 70000.0 1e-7 0.1])))
  (is (= [1.0 1.0009765625 5.960464477539063E-8 65504.0] (map #(double (t/half->float %)) [15360 0x3c01 1 0x7bff])))
  (is (Float/isInfinite (t/half->float 0x7c00)))
  (is (Float/isNaN (t/half->float 0x7e00)))
  (testing "every finite half survives a round trip"
    (is (every? (fn [h] (= h (t/float->half (t/half->float h))))
                (concat (range 0 0x7c00 7) (range 0x8000 0xfc00 11))))))

(deftest erf-and-gelu
  (is (< (Math/abs (- (t/erf 0.5) 0.5204998778130465)) 1e-14))
  (is (< (Math/abs (- (t/erf -2.0) -0.9953222650189527)) 1e-14))
  (is (h/approx= (vec (t/gelu! (float-array [-1.0 0.0 1.0 2.0])))
                 [-0.15865525393145707 0.0 0.8413447460685429 1.9544997361036416] 1e-6)))
