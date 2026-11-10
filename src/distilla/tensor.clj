(ns distilla.tensor
  "Row-major float32 matrices on the jolt heap and the element-wise kernels
  the memory module needs: layer norm, GELU, ReLU, masked softmax, L2
  normalisation, plus float16 / bfloat16 conversion.

  A matrix is {:rows r :cols c :data float-array}; a vector is a float array."
  (:refer-clojure :exclude [vector]))

(defn matrix
  ([rows cols] {:rows rows :cols cols :data (float-array (* rows cols))})
  ([rows cols ^floats data]
   (when-not (= (alength data) (* rows cols))
     (throw (ex-info "matrix data does not match its shape" {:rows rows :cols cols :length (alength data)})))
   {:rows rows :cols cols :data data}))

(defn from-rows
  "A matrix from a sequence of equal-length number sequences (or float arrays).
  cols must be given when there are no rows."
  ([rows] (from-rows rows nil))
  ([rows cols]
   (let [rows (vec rows)
         c (or cols (if (seq rows) (count (first rows)) 0))
         out (float-array (* (count rows) c))]
     (dotimes [i (count rows)]
       (let [row (nth rows i)]
         (when-not (= c (count row))
           (throw (ex-info "rows differ in length" {:expected c :got (count row)})))
         (if (instance? (class (float-array 0)) row)
           (System/arraycopy row 0 out (* i c) c)
           (loop [j 0 xs (seq row)]
             (when xs
               (aset out (+ (* i c) j) (float (first xs)))
               (recur (inc j) (next xs)))))))
     {:rows (count rows) :cols c :data out})))

(defn row
  "Row i of a matrix as a new float array."
  [{:keys [cols data]} i]
  (let [out (float-array cols)]
    (System/arraycopy data (* i cols) out 0 cols)
    out))

(defn rows->vecs
  "A matrix as a vector of vectors of doubles (for tests and JSON)."
  [{:keys [rows cols data]}]
  (let [^floats data data]
    (vec (for [i (range rows)]
           (vec (for [j (range cols)] (double (aget data (+ (* i cols) j)))))))))

(defn select-rows
  "The rows of m at the given indices, as a new matrix."
  [{:keys [cols data]} indices]
  (let [indices (vec indices)
        out (float-array (* (count indices) cols))]
    (dotimes [r (count indices)]
      (System/arraycopy data (* (nth indices r) cols) out (* r cols) cols))
    {:rows (count indices) :cols cols :data out}))

(defn take-rows
  "The first n rows of m."
  [m n]
  (select-rows m (range n)))

(defn concat-rows
  "Stack matrices of equal width."
  [ms cols]
  (let [total (reduce + 0 (map :rows ms))
        out (float-array (* total cols))]
    (loop [ms ms offset 0]
      (when-let [m (first ms)]
        (System/arraycopy (:data m) 0 out offset (* (:rows m) cols))
        (recur (rest ms) (+ offset (* (:rows m) cols)))))
    {:rows total :cols cols :data out}))

(defn copy-floats ^floats [^floats a]
  (let [out (float-array (alength a))]
    (System/arraycopy a 0 out 0 (alength a))
    out))

(defn add!
  "a += b element-wise; returns a."
  [^floats a ^floats b]
  (dotimes [i (alength a)]
    (aset a i (float (+ (aget a i) (aget b i)))))
  a)

(defn add-bias!
  "Add a bias vector to every row of an (n x c) float array; returns it."
  [^floats a cols ^floats bias]
  (when bias
    (let [n (quot (alength a) cols)]
      (dotimes [i n]
        (let [base (* i cols)]
          (dotimes [j cols]
            (aset a (+ base j) (float (+ (aget a (+ base j)) (aget bias j)))))))))
  a)

(defn layer-norm
  "Row-wise LayerNorm of an (n x c) float array with weight, bias and eps
  (torch's default 1e-5). Returns a new array."
  ([a cols weight bias] (layer-norm a cols weight bias 1e-5))
  ([^floats a cols ^floats weight ^floats bias eps]
   (let [n (quot (alength a) cols)
         out (float-array (alength a))]
     (dotimes [i n]
       (let [base (* i cols)
             mean (/ (loop [j 0 s 0.0]
                       (if (< j cols) (recur (inc j) (+ s (aget a (+ base j)))) s))
                     cols)
             var (/ (loop [j 0 s 0.0]
                      (if (< j cols)
                        (let [d (- (aget a (+ base j)) mean)] (recur (inc j) (+ s (* d d))))
                        s))
                    cols)
             inv (/ 1.0 (Math/sqrt (+ var eps)))]
         (dotimes [j cols]
           (aset out (+ base j)
                 (float (+ (* (- (aget a (+ base j)) mean) inv (aget weight j))
                           (aget bias j)))))))
     out)))

(defn erf
  "The error function, through the Chebyshev fit of erfc from Numerical
  Recipes (3rd ed., Erf::erfccheb), accurate to about 1e-16."
  ^double [^double x]
  (let [z (Math/abs x)
        t (/ 2.0 (+ 2.0 z))
        ty (- (* 4.0 t) 2.0)
        cof [-1.3026537197817094 6.4196979235649026e-1 1.9476473204185836e-2 -9.561514786808631e-3
             -9.46595344482036e-4 3.66839497852761e-4 4.2523324806907e-5 -2.0278578112534e-5
             -1.624290004647e-6 1.303655835580e-6 1.5626441722e-8 -8.5238095915e-8
             6.529054439e-9 5.059343495e-9 -9.91364156e-10 -2.27365122e-10
             9.6467911e-11 2.394038e-12 -6.886027e-12 8.94487e-13 3.13092e-13
             -1.12708e-13 3.81e-16 7.106e-15 -1.523e-15 -9.4e-17 1.21e-16 -2.8e-17]
        [d dd] (loop [j (dec (count cof)) d 0.0 dd 0.0]
                 (if (pos? j)
                   (recur (dec j) (+ (- (* ty d) dd) (nth cof j)) d)
                   [d dd]))
        erfc (* t (Math/exp (+ (- (* z z)) (* 0.5 (+ (nth cof 0) (* ty d))) (- dd))))]
    (if (>= x 0.0) (- 1.0 erfc) (- erfc 1.0))))

(defn gelu!
  "Exact (erf) GELU in place, torch's nn.GELU() default."
  [^floats a]
  (let [inv-sqrt2 (/ 1.0 (Math/sqrt 2.0))]
    (dotimes [i (alength a)]
      (let [x (double (aget a i))]
        (aset a i (float (* 0.5 x (+ 1.0 (erf (* x inv-sqrt2)))))))))
  a)

(defn relu! [^floats a]
  (dotimes [i (alength a)]
    (when (< (aget a i) 0.0) (aset a i (float 0.0))))
  a)

(defn softmax-rows!
  "Softmax of each row of an (n x c) float array in place. mask, when given,
  is a boolean sequence over the c columns, true at positions to ignore."
  [^floats a cols mask]
  (let [n (quot (alength a) cols)
        masked (when mask (boolean-array (map boolean mask)))]
    (dotimes [i n]
      (let [base (* i cols)
            mx (loop [j 0 m Double/NEGATIVE_INFINITY]
                 (if (< j cols)
                   (recur (inc j) (if (and masked (aget masked j)) m (max m (aget a (+ base j)))))
                   m))
            total (loop [j 0 s 0.0]
                    (if (< j cols)
                      (if (and masked (aget masked j))
                        (do (aset a (+ base j) (float 0.0)) (recur (inc j) s))
                        (let [e (Math/exp (- (aget a (+ base j)) mx))]
                          (aset a (+ base j) (float e))
                          (recur (inc j) (+ s e))))
                      s))]
        (dotimes [j cols]
          (aset a (+ base j) (float (/ (aget a (+ base j)) total)))))))
  a)

(defn l2-normalize-rows
  "Each row divided by max(norm, eps), like torch.nn.functional.normalize."
  ([a cols] (l2-normalize-rows a cols 1e-12))
  ([^floats a cols eps]
   (let [n (quot (alength a) cols)
         out (float-array (alength a))]
     (dotimes [i n]
       (let [base (* i cols)
             norm (Math/sqrt (loop [j 0 s 0.0]
                               (if (< j cols)
                                 (let [v (aget a (+ base j))] (recur (inc j) (+ s (* v v))))
                                 s)))
             d (max norm eps)]
         (dotimes [j cols]
           (aset out (+ base j) (float (/ (aget a (+ base j)) d))))))
     out)))

(defn dot ^double [^floats a ^floats b]
  (loop [i 0 s 0.0]
    (if (< i (alength a)) (recur (inc i) (+ s (* (aget a i) (aget b i)))) s)))

(defn norm ^double [^floats a] (Math/sqrt (dot a a)))

;; --- half precision ------------------------------------------------------------

(defn half->float
  "IEEE binary16 bits (0..65535) to a float."
  [h]
  (let [h (bit-and (long h) 0xffff)
        sign (if (zero? (bit-and h 0x8000)) 1.0 -1.0)
        exp (bit-and (bit-shift-right h 10) 0x1f)
        mant (bit-and h 0x3ff)]
    (cond
      (zero? exp) (float (* sign (Math/pow 2.0 -14) (/ mant 1024.0)))
      (= exp 31) (Float/intBitsToFloat (unchecked-int (bit-or (if (neg? sign) 0x80000000 0)
                                                               0x7f800000
                                                               (bit-shift-left mant 13))))
      :else (float (* sign (Math/pow 2.0 (- exp 15)) (+ 1.0 (/ mant 1024.0)))))))

(def ^:private half-table
  (delay (let [t (float-array 65536)]
           (dotimes [i 65536] (aset t i (float (half->float i))))
           t)))

(defn halves->floats
  "A short array of binary16 bits to a float array."
  [^shorts s]
  (let [^floats table @half-table
        out (float-array (alength s))]
    (dotimes [i (alength s)]
      (aset out i (aget table (bit-and (aget s i) 0xffff))))
    out))

(defn float->half
  "Round a float to IEEE binary16 bits (round to nearest even), as a long 0..65535."
  [x]
  (let [bits (Float/floatToRawIntBits (float x))
        sign (bit-and (unsigned-bit-shift-right bits 16) 0x8000)
        exp (bit-and (unsigned-bit-shift-right bits 23) 0xff)
        mant (bit-and bits 0x7fffff)]
    (cond
      (= exp 255) (bit-or sign 0x7c00 (if (zero? mant) 0 0x200))
      :else
      (let [e (- exp 112)]                    ; rebias 127 -> 15
        (cond
          (>= e 31) (bit-or sign 0x7c00)
          (<= e 0)
          (if (< e -10)
            sign
            (let [m (bit-or mant 0x800000)
                  shift (- 14 e)
                  half-m (unsigned-bit-shift-right m shift)
                  rem (bit-and m (dec (bit-shift-left 1 shift)))
                  halfway (bit-shift-left 1 (dec shift))
                  half-m (if (or (> rem halfway) (and (= rem halfway) (odd? half-m))) (inc half-m) half-m)]
              (bit-or sign half-m)))
          :else
          (let [half-m (unsigned-bit-shift-right mant 13)
                rem (bit-and mant 0x1fff)
                v (bit-or (bit-shift-left e 10) half-m)
                v (if (or (> rem 0x1000) (and (= rem 0x1000) (odd? half-m))) (inc v) v)]
            (if (>= v 0x7c00) (bit-or sign 0x7c00) (bit-or sign v))))))))

(defn floats->halves
  "A float array to a short array of binary16 bits."
  [^floats a]
  (let [out (short-array (alength a))]
    (dotimes [i (alength a)]
      (aset out i (unchecked-short (float->half (aget a i)))))
    out))

(defn round-to-half
  "Round every element to the nearest float16 value (what storing as float16 does)."
  [^floats a]
  (halves->floats (floats->halves a)))

(defn bf16s->floats
  "A short array of bfloat16 bits to a float array."
  [^shorts s]
  (let [out (float-array (alength s))]
    (dotimes [i (alength s)]
      (aset out i (Float/intBitsToFloat (unchecked-int (bit-shift-left (bit-and (aget s i) 0xffff) 16)))))
    out))

(defn float->bf16
  "Round a float to bfloat16 bits (round to nearest even)."
  [x]
  (let [bits (bit-and (Float/floatToRawIntBits (float x)) 0xffffffff)]
    (if (Float/isNaN (float x))
      0x7fc0
      (bit-and (unsigned-bit-shift-right (+ bits 0x7fff (bit-and (unsigned-bit-shift-right bits 16) 1)) 16)
               0xffff))))

(defn floats->bf16s [^floats a]
  (let [out (short-array (alength a))]
    (dotimes [i (alength a)]
      (aset out i (unchecked-short (float->bf16 (aget a i)))))
    out))
