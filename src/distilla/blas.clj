(ns distilla.blas
  "Single-precision matrix products through CBLAS (OpenBLAS on Linux,
  Accelerate on macOS), on foreign memory.

  Weight matrices live in foreign memory for the life of a model; activations
  are float arrays on the jolt heap, copied into a scratch arena around each
  product. All matrices are row-major."
  (:require [jolt.ffi :as ffi]))

(def ^:const row-major 101)
(def ^:const no-trans 111)
(def ^:const trans 112)

(ffi/defcfn cblas-sgemm "cblas_sgemm"
  [:int :int :int :int :int :int :float :pointer :int :pointer :int :float :pointer :int] :void :blocking)

(defn gemm!
  "C = alpha * op(A) * op(B) + beta * C on foreign pointers (row-major).
  op(A) is m x k, op(B) is k x n, C is m x n."
  [trans-a? trans-b? m n k alpha a lda b ldb beta c ldc]
  (when (and (pos? m) (pos? n))
    (cblas-sgemm row-major (if trans-a? trans no-trans) (if trans-b? trans no-trans)
                 m n k (float alpha) a lda b ldb (float beta) c ldc)))

(defn ->foreign
  "Copy a float array into foreign memory owned by arena."
  [arena ^floats data]
  (let [n (alength data)
        p (ffi/alloc arena (* 4 (max 1 n)))]
    (when (pos? n) (ffi/write-array p :float data))
    p))

(defn foreign->floats
  "Read n floats from foreign memory into a new float array."
  [p n]
  (if (pos? n) (ffi/read-array p :float n) (float-array 0)))

(defn alloc-floats [arena n]
  (ffi/alloc arena (* 4 (max 1 n))))

(defn matmul-t
  "X (m x k float array) times the transpose of the foreign weight W
  (n x k, the layout of a torch Linear weight): an m x n float array."
  [x m k w n]
  (with-open [arena (ffi/confined-arena)]
    (let [xp (->foreign arena x)
          yp (alloc-floats arena (* m n))]
      (gemm! false true m n k 1.0 xp k w k 0.0 yp n)
      (foreign->floats yp (* m n)))))
