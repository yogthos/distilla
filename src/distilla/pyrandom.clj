(ns distilla.pyrandom
  "Bit-exact ports of the two seeded generators the Python pipelines use, so
  datasets built here match the released ones:

  - Python's random.Random (MT19937): seed, random, getrandbits, randrange,
    randint, choice, shuffle.
  - numpy's default_rng(seed).permutation(n) (SeedSequence -> PCG64 ->
    bounded interval draws).")

;; --- Python random.Random -------------------------------------------------------------------

(def ^:private mt-n 624)
(def ^:private mt-m 397)

(defn- u32 [x] (bit-and (long x) 0xffffffff))

(defn- init-genrand [^longs mt s]
  (aset mt 0 (u32 s))
  (loop [i 1]
    (when (< i mt-n)
      (let [prev (aget mt (dec i))]
        (aset mt i (u32 (+ (* 1812433253 (bit-xor prev (unsigned-bit-shift-right prev 30))) i))))
      (recur (inc i)))))

(defn- init-by-array [^longs mt key]
  (init-genrand mt 19650218)
  (let [klen (count key)
        [i j] (loop [k (max mt-n klen) i 1 j 0]
                (if (zero? k)
                  [i j]
                  (let [prev (aget mt (dec i))
                        v (u32 (+ (bit-xor (aget mt i) (* (bit-xor prev (unsigned-bit-shift-right prev 30)) 1664525))
                                  (nth key j) j))]
                    (aset mt i v)
                    (let [i (inc i) j (inc j)
                          i (if (>= i mt-n) (do (aset mt 0 (aget mt (dec mt-n))) 1) i)
                          j (if (>= j klen) 0 j)]
                      (recur (dec k) i j)))))]
    (loop [k (dec mt-n) i i]
      (when (pos? k)
        (let [prev (aget mt (dec i))]
          (aset mt i (u32 (- (bit-xor (aget mt i) (* (bit-xor prev (unsigned-bit-shift-right prev 30)) 1566083941)) i))))
        (let [i (inc i)
              i (if (>= i mt-n) (do (aset mt 0 (aget mt (dec mt-n))) 1) i)]
          (recur (dec k) i))))
    (aset mt 0 0x80000000)))

(defn random-state
  "A random.Random(seed) for an integer seed (Python takes its absolute value)."
  [seed]
  (let [seed (Math/abs (long seed))
        key (if (zero? seed)
              [0]
              (loop [s seed acc []]
                (if (zero? s)
                  acc
                  (recur (unsigned-bit-shift-right s 32) (conj acc (bit-and s 0xffffffff))))))
        mt (long-array mt-n)]
    (init-by-array mt key)
    (atom {:mt mt :index mt-n})))

(defn- twist! [^longs mt]
  (dotimes [kk mt-n]
    (let [y (bit-or (bit-and (aget mt kk) 0x80000000) (bit-and (aget mt (mod (inc kk) mt-n)) 0x7fffffff))
          v (bit-xor (aget mt (mod (+ kk mt-m) mt-n)) (unsigned-bit-shift-right y 1) (if (odd? y) 0x9908b0df 0))]
      (aset mt kk v))))

(defn genrand-uint32
  "The next 32-bit output."
  [state]
  (let [{:keys [^longs mt index]} @state
        index (if (>= index mt-n) (do (twist! mt) 0) index)
        y (aget mt index)
        y (bit-xor y (unsigned-bit-shift-right y 11))
        y (bit-xor y (bit-and (bit-shift-left y 7) 0x9d2c5680))
        y (bit-xor y (bit-and (bit-shift-left y 15) 0xefc60000))
        y (bit-xor y (unsigned-bit-shift-right y 18))]
    (swap! state assoc :index (inc index))
    (u32 y)))

(defn random
  "random.random(): a float in [0, 1) from 53 bits."
  ^double [state]
  (let [a (unsigned-bit-shift-right (genrand-uint32 state) 5)
        b (unsigned-bit-shift-right (genrand-uint32 state) 6)]
    (* (+ (* a 67108864.0) b) (/ 1.0 9007199254740992.0))))

(defn getrandbits
  "random.getrandbits(k) for 1 <= k <= 63, as a long."
  [state k]
  (when-not (<= 1 k 63) (throw (ex-info "getrandbits supports 1..63 bits here" {:k k})))
  (if (<= k 32)
    (unsigned-bit-shift-right (genrand-uint32 state) (- 32 k))
    (let [lo (genrand-uint32 state)
          hi (unsigned-bit-shift-right (genrand-uint32 state) (- 64 k))]
      (bit-or lo (bit-shift-left hi 32)))))

(defn- bit-length [n] (- 64 (Long/numberOfLeadingZeros (long n))))

(defn randbelow
  "random._randbelow(n): uniform in [0, n) by rejection on bit_length(n) bits."
  [state n]
  (let [k (bit-length n)]
    (loop []
      (let [r (getrandbits state k)]
        (if (>= r n) (recur) r)))))

(defn randrange
  ([state stop] (randbelow state stop))
  ([state start stop] (+ start (randbelow state (- stop start)))))

(defn randint [state a b] (randrange state a (inc b)))

(defn choice [state xs] (nth (vec xs) (randbelow state (count xs))))

(defn shuffle
  "random.shuffle on a vector: a new vector."
  [state xs]
  (let [a (object-array xs)]
    (loop [i (dec (alength a))]
      (when (pos? i)
        (let [j (randbelow state (inc i))
              t (aget a i)]
          (aset a i (aget a j))
          (aset a j t))
        (recur (dec i))))
    (vec a)))

;; --- numpy default_rng ------------------------------------------------------------------------
;;
;; PCG64 works on 128-bit integers; here they are vectors of eight 16-bit
;; limbs (least significant first), so every partial product fits a long.

(def ^:private mask32 0xffffffff)

(defn- limbs [hi64 lo64]
  (vec (for [i (range 8)]
         (let [w (if (< i 4) lo64 hi64)]
           (bit-and (unsigned-bit-shift-right w (* 16 (mod i 4))) 0xffff)))))

(defn- limbs->u64 [l k] ; k = 0 low half, 1 high half
  (reduce (fn [acc i] (bit-or acc (bit-shift-left (nth l (+ (* 4 k) i)) (* 16 i)))) 0 (range 4)))

(defn- add128 [a b]
  (loop [i 0 carry 0 out []]
    (if (= i 8)
      out
      (let [s (+ (nth a i) (nth b i) carry)]
        (recur (inc i) (unsigned-bit-shift-right s 16) (conj out (bit-and s 0xffff)))))))

(defn- mul128 [a b]
  (let [cols (long-array 8)]
    (dotimes [i 8]
      (dotimes [j (- 8 i)]
        (aset cols (+ i j) (+ (aget cols (+ i j)) (* (nth a i) (nth b j))))))
    (loop [i 0 carry 0 out []]
      (if (= i 8)
        out
        (let [s (+ (aget cols i) carry)]
          (recur (inc i) (unsigned-bit-shift-right s 16) (conj out (bit-and s 0xffff))))))))

(defn- shl1-or1 [a] ; (a << 1) | 1, mod 2^128
  (loop [i 0 carry 1 out []]
    (if (= i 8)
      out
      (let [s (bit-or (bit-shift-left (nth a i) 1) carry)]
        (recur (inc i) (unsigned-bit-shift-right s 16) (conj out (bit-and s 0xffff)))))))

(def ^:private pcg-mult (limbs 0x2360ed051fc65da4 0x4385df649fccf645))

(defn- umul32 [a b] (bit-and (unchecked-multiply (long a) (long b)) mask32))

(defn- seed-sequence-state
  "numpy SeedSequence(entropy).generate_state(n-words, uint32), entropy a
  non-negative integer below 2^32."
  [entropy n-words]
  (let [hash-const (atom 0x43b0d7e5)
        hashmix (fn [value]
                  (let [value (bit-xor (u32 value) @hash-const)
                        _ (swap! hash-const #(umul32 % 0x931e8875))
                        value (umul32 value @hash-const)]
                    (bit-xor value (unsigned-bit-shift-right value 16))))
        mix (fn [x y]
              (let [r (bit-and (- (umul32 0xca01f9dd x) (umul32 0x4973f715 y)) mask32)]
                (bit-xor r (unsigned-bit-shift-right r 16))))
        entropy [(u32 entropy)]
        pool (long-array 4)]
    (dotimes [i 4] (aset pool i (hashmix (if (< i (count entropy)) (nth entropy i) 0))))
    (dotimes [src 4]
      (dotimes [dst 4]
        (when-not (= src dst)
          (aset pool dst (mix (aget pool dst) (hashmix (aget pool src)))))))
    (let [hc (atom 0x8b51f9dd)]
      (vec (for [i (range n-words)]
             (let [v (bit-xor (aget pool (mod i 4)) @hc)
                   _ (swap! hc #(umul32 % 0x58f38ded))
                   v (umul32 v @hc)]
               (bit-xor v (unsigned-bit-shift-right v 16))))))))

(defn- pcg-step [state inc] (add128 (mul128 state pcg-mult) inc))

(defn default-rng
  "numpy.random.default_rng(seed) for an integer seed in [0, 2^32): a PCG64 state."
  [seed]
  (let [w (seed-sequence-state seed 8)
        u64 (fn [k] (bit-or (nth w (* 2 k)) (bit-shift-left (nth w (inc (* 2 k))) 32)))
        initstate (limbs (u64 0) (u64 1))
        initseq (limbs (u64 2) (u64 3))
        inc (shl1-or1 initseq)
        state (pcg-step (vec (repeat 8 0)) inc)
        state (pcg-step (add128 state initstate) inc)]
    (atom {:state state :inc inc :has-uint32 false :uint32 0})))

(defn next-uint64
  "The next 64-bit output (XSL-RR), as a long holding the unsigned bits."
  [rng]
  (let [{:keys [state inc]} @rng
        state (pcg-step state inc)
        x (bit-xor (limbs->u64 state 1) (limbs->u64 state 0))
        rot (unsigned-bit-shift-right (nth state 7) 10)]
    (swap! rng assoc :state state)
    (if (zero? rot) x (bit-or (unsigned-bit-shift-right x rot) (bit-shift-left x (- 64 rot))))))

(defn next-uint32 [rng]
  (if (:has-uint32 @rng)
    (let [v (:uint32 @rng)] (swap! rng assoc :has-uint32 false) v)
    (let [v (next-uint64 rng)]
      (swap! rng assoc :has-uint32 true :uint32 (unsigned-bit-shift-right v 32))
      (bit-and v mask32))))

(defn random-interval
  "numpy's random_interval(max): uniform in [0, max] by masked rejection."
  [rng max]
  (if (zero? max)
    0
    (let [mask (loop [m max s 1] (if (> s 32) m (recur (bit-or m (unsigned-bit-shift-right m s)) (* 2 s))))]
      (if (<= max mask32)
        (loop [] (let [v (bit-and (next-uint32 rng) mask)] (if (> v max) (recur) v)))
        (loop [] (let [v (bit-and (next-uint64 rng) mask)] (if (> v max) (recur) v)))))))

(defn permutation
  "numpy default_rng(seed).permutation(n) as a vector."
  [rng n]
  (let [a (long-array (range n))]
    (loop [i (dec n)]
      (when (pos? i)
        (let [j (random-interval rng i)
              t (aget a i)]
          (aset a i (aget a j))
          (aset a j t))
        (recur (dec i))))
    (vec a)))
