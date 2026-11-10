(ns distilla.torch
  "Read PyTorch checkpoints (torch.save's zip format) without Python.

  A .pt file is an uncompressed zip holding <prefix>/data.pkl, a pickle of
  the saved object, and <prefix>/data/<key>, one raw little-endian file per
  storage. The pickle names storages through persistent ids and rebuilds
  tensors with torch._utils._rebuild_tensor_v2. This reader runs the subset
  of the pickle machine that state dicts use and returns plain data: maps
  for dicts and OrderedDicts, vectors for lists and tuples, and tensors as
  {:dtype :shape :data} with float data widened to a float array (row-major,
  honouring storage offsets and strides)."
  (:require [clojure.string :as str]
            [distilla.io :as io]
            [distilla.tensor :as t]))

;; --- zip ---------------------------------------------------------------------

(defn- u16 [^bytes b off] (bit-or (bit-and (aget b off) 0xff) (bit-shift-left (bit-and (aget b (inc off)) 0xff) 8)))
(defn- u32 [^bytes b off] (bit-or (u16 b off) (bit-shift-left (u16 b (+ off 2)) 16)))
(defn- u64 [^bytes b off] (bit-or (u32 b off) (bit-shift-left (u32 b (+ off 4)) 32)))

(defn- zip64-extra
  "Values from a zip64 extended-information extra field, in order."
  [^bytes extra]
  (loop [off 0]
    (when (< (+ off 4) (alength extra))
      (let [id (u16 extra off) size (u16 extra (+ off 2))]
        (if (= id 1)
          (vec (for [i (range (quot size 8))] (u64 extra (+ off 4 (* 8 i)))))
          (recur (+ off 4 size)))))))

(defn zip-entries
  "{name {:offset data-offset :size n :method m}} for every entry of a zip."
  [path]
  (let [size (io/file-size path)
        tail-len (min size (+ 22 65535))
        tail (io/read-bytes path (- size tail-len) tail-len)
        eocd (loop [i (- tail-len 22)]
               (cond (neg? i) (throw (ex-info "not a zip file" {:path path}))
                     (= (u32 tail i) 0x06054b50) i
                     :else (recur (dec i))))
        n-entries (u16 tail (+ eocd 10))
        cd-size (u32 tail (+ eocd 12))
        cd-offset (u32 tail (+ eocd 16))
        ;; zip64: the locator sits just before the end record
        [n-entries cd-size cd-offset]
        (if (and (>= eocd 20) (= (u32 tail (- eocd 20)) 0x07064b50))
          (let [rec-off (u64 tail (- eocd 12))
                rec (io/read-bytes path rec-off 56)]
            [(u64 rec 32) (u64 rec 40) (u64 rec 48)])
          [n-entries cd-size cd-offset])
        cd (io/read-bytes path cd-offset cd-size)]
    (loop [i 0 off 0 acc {}]
      (if (= i n-entries)
        acc
        (let [method (u16 cd (+ off 10))
              csize (u32 cd (+ off 20))
              usize (u32 cd (+ off 24))
              name-len (u16 cd (+ off 28))
              extra-len (u16 cd (+ off 30))
              comment-len (u16 cd (+ off 32))
              local (u32 cd (+ off 42))
              name (String. ^bytes (java.util.Arrays/copyOfRange cd (+ off 46) (+ off 46 name-len)) "UTF-8")
              extra (java.util.Arrays/copyOfRange cd (+ off 46 name-len) (+ off 46 name-len extra-len))
              ;; zip64 fields replace whichever of these hold 0xffffffff, in this order
              z64 (zip64-extra extra)
              [usize csize local] (loop [vals [usize csize local] k 0 z z64 out []]
                                    (if (= k 3)
                                      out
                                      (let [v (nth vals k)]
                                        (if (and (= v 0xffffffff) (seq z))
                                          (recur vals (inc k) (rest z) (conj out (first z)))
                                          (recur vals (inc k) z (conj out v))))))
              lh (io/read-bytes path local 30)
              data-off (+ local 30 (u16 lh 26) (u16 lh 28))]
          (recur (inc i) (+ off 46 name-len extra-len comment-len)
                 (assoc acc name {:offset data-off :size usize :csize csize :method method})))))))

;; --- pickle ------------------------------------------------------------------

(def ^:private storage-dtypes
  {"FloatStorage" ["F32" :float 4]
   "HalfStorage" ["F16" :uint16 2]
   "BFloat16Storage" ["BF16" :uint16 2]
   "DoubleStorage" ["F64" :double 8]
   "LongStorage" ["I64" :int64 8]
   "IntStorage" ["I32" :int32 4]
   "ShortStorage" ["I16" :int16 2]
   "ByteStorage" ["U8" :uint8 1]
   "CharStorage" ["I8" :uint8 1]
   "BoolStorage" ["BOOL" :uint8 1]})

(defrecord Global [module name])
(defrecord Storage [dtype io-type width key numel])
(defrecord Mark [])

(defn- read-storage-data [path entries prefix {:keys [io-type key numel]}]
  (let [entry (or (get entries (str prefix "data/" key))
                  (throw (ex-info (str "storage " key " missing from checkpoint") {:key key})))]
    (when-not (zero? (:method entry))
      (throw (ex-info "compressed checkpoint entries are not supported" {:entry entry})))
    (io/read-array path (:offset entry) io-type numel)))

(defn- widen [dtype arr]
  (case dtype
    "F32" arr
    "F16" (t/halves->floats arr)
    "BF16" (t/bf16s->floats arr)
    "F64" (float-array (alength ^doubles arr) (seq arr))
    ("I64") arr
    ("I32" "I16") (long-array arr)
    arr))

(defn- strided-copy
  "Gather a tensor of shape/stride starting at offset out of a flat storage array."
  [src offset shape stride]
  (let [n (reduce * 1 shape)
        contiguous? (= stride (vec (reverse (reductions * 1 (reverse (rest shape))))))
        float? (instance? (class (float-array 0)) src)]
    (if (and contiguous? (zero? offset) (= n (alength src)))
      src
      (let [out (if float? (float-array n) (long-array n))
            dims (count shape)]
        (dotimes [i n]
          (let [src-index (loop [d (dec dims) rem i acc offset]
                            (if (neg? d)
                              acc
                              (let [size (nth shape d)]
                                (recur (dec d) (quot rem size) (+ acc (* (mod rem size) (nth stride d)))))))]
            (if float?
              (aset ^floats out i (aget ^floats src src-index))
              (aset ^longs out i (long (aget src src-index))))))
        out))))

(def ^:dynamic *lazy*
  "When true, tensors are read on demand: :data is a delay."
  false)

(defn- rebuild-tensor [load-storage storage offset shape stride]
  (let [shape (vec shape)
        stride (vec stride)
        read (fn [] (strided-copy (widen (:dtype storage) (load-storage storage)) offset shape stride))]
    {:dtype (:dtype storage) :shape shape :data (if *lazy* (delay (read)) (read))}))

(defn- call-global [load-storage {:keys [module name] :as g} args]
  (case [module name]
    ["collections" "OrderedDict"] (if (seq args) (into {} (first args)) {})
    ["torch._utils" "_rebuild_tensor_v2"] (let [[storage offset shape stride] args]
                                            (rebuild-tensor load-storage storage offset shape stride))
    ["torch._utils" "_rebuild_tensor"] (let [[storage offset shape stride] args]
                                         (rebuild-tensor load-storage storage offset shape stride))
    ["torch._utils" "_rebuild_parameter"] (first args)
    ["torch._utils" "_rebuild_parameter_with_state"] (first args)
    ;; a Tensor subclass: rebuild the tensor, drop the subclass
    ["torch._tensor" "_rebuild_from_type_v2"] (let [[f _type inner-args _state] args]
                                                (call-global load-storage f inner-args))
    ["torch" "Size"] (vec (first args))
    ["builtins" "set"] (set (first args))
    ["__builtin__" "set"] (set (first args))
    (throw (ex-info (str "unsupported pickle global " module "." name) {:global g}))))

(defn- persistent-load [pid]
  ;; ('storage', storage_type, key, location, numel)
  (let [[tag storage-type key _location numel] pid]
    (when-not (= tag "storage")
      (throw (ex-info "unsupported persistent id" {:pid pid})))
    (let [type-name (:name storage-type)
          [dtype io-type width] (or (storage-dtypes type-name)
                                    (throw (ex-info (str "unsupported storage " type-name) {:pid pid})))]
      (->Storage dtype io-type width (str key) numel))))

(defn unpickle
  "Run a pickle (protocol <= 4 subset) held in bytes. load-storage reads a
  Storage record's raw data."
  [^bytes b load-storage]
  (let [n (alength b)
        u8 (fn [i] (bit-and (aget b i) 0xff))
        i32 (fn [i] (let [v (u32 b i)] (if (>= v 0x80000000) (- v 0x100000000) v)))
        string (fn [start len] (String. ^bytes (java.util.Arrays/copyOfRange b (int start) (int (+ start len))) "UTF-8"))
        line (fn [start] (loop [j start] (if (= 10 (u8 j)) j (recur (inc j)))))
        pop-mark (fn [stack]
                   (let [idx (loop [j (dec (count stack))]
                               (if (instance? Mark (nth stack j)) j (recur (dec j))))]
                     [(subvec stack 0 idx) (subvec stack (inc idx))]))
        set-items (fn [d items]
                    (reduce (fn [m [k v]] (assoc m k v)) d (partition 2 items)))
        signed-le (fn [start len]
                    (if (zero? len)
                      0
                      (let [v (loop [j (dec len) acc (biginteger 0)]
                                (if (neg? j)
                                  acc
                                  (recur (dec j) (.add (.shiftLeft acc 8) (biginteger (u8 (+ start j)))))))
                            v (if (>= (u8 (+ start len -1)) 0x80)
                                (.subtract v (.shiftLeft (biginteger 1) (* 8 len)))
                                v)]
                        (long v))))]
    (loop [pc 0 stack [] memo {}]
      (when (>= pc n) (throw (ex-info "pickle ended without STOP" {})))
      (let [op (u8 pc) pc (inc pc)
            top (when (seq stack) (peek stack))
            pop1 (when (seq stack) (pop stack))]
        (case (int op)
          0x80 (recur (inc pc) stack memo)                                    ; PROTO
          0x95 (recur (+ pc 8) stack memo)                                    ; FRAME
          0x2e top                                                            ; STOP
          0x7d (recur pc (conj stack {}) memo)                                ; EMPTY_DICT
          0x5d (recur pc (conj stack []) memo)                                ; EMPTY_LIST
          0x29 (recur pc (conj stack []) memo)                                ; EMPTY_TUPLE
          0x28 (recur pc (conj stack (->Mark)) memo)                          ; MARK
          0x4e (recur pc (conj stack nil) memo)                               ; NONE
          0x88 (recur pc (conj stack true) memo)                              ; NEWTRUE
          0x89 (recur pc (conj stack false) memo)                             ; NEWFALSE
          0x4b (recur (inc pc) (conj stack (u8 pc)) memo)                     ; BININT1
          0x4d (recur (+ pc 2) (conj stack (u16 b pc)) memo)                  ; BININT2
          0x4a (recur (+ pc 4) (conj stack (i32 pc)) memo)                    ; BININT
          0x8a (let [len (u8 pc)]                                             ; LONG1
                 (recur (+ pc 1 len) (conj stack (signed-le (inc pc) len)) memo))
          0x47 (recur (+ pc 8) (conj stack (Double/longBitsToDouble        ; BINFLOAT (big-endian)
                                            (loop [j 0 acc 0] (if (= j 8) acc (recur (inc j) (bit-or (bit-shift-left acc 8) (u8 (+ pc j))))))))
                      memo)
          0x58 (let [len (u32 b pc)]                                          ; BINUNICODE
                 (recur (+ pc 4 len) (conj stack (string (+ pc 4) len)) memo))
          0x8c (let [len (u8 pc)]                                             ; SHORT_BINUNICODE
                 (recur (+ pc 1 len) (conj stack (string (inc pc) len)) memo))
          0x8d (let [len (u64 b pc)]                                          ; BINUNICODE8
                 (recur (+ pc 8 len) (conj stack (string (+ pc 8) len)) memo))
          0x55 (let [len (u8 pc)]                                             ; SHORT_BINSTRING
                 (recur (+ pc 1 len) (conj stack (string (inc pc) len)) memo))
          0x54 (let [len (u32 b pc)]                                          ; BINSTRING
                 (recur (+ pc 4 len) (conj stack (string (+ pc 4) len)) memo))
          0x43 (let [len (u8 pc)]                                             ; SHORT_BINBYTES
                 (recur (+ pc 1 len) (conj stack (java.util.Arrays/copyOfRange b (int (inc pc)) (int (+ pc 1 len)))) memo))
          0x42 (let [len (u32 b pc)]                                          ; BINBYTES
                 (recur (+ pc 4 len) (conj stack (java.util.Arrays/copyOfRange b (int (+ pc 4)) (int (+ pc 4 len)))) memo))
          0x63 (let [e1 (line pc) e2 (line (inc e1))]                         ; GLOBAL
                 (recur (inc e2) (conj stack (->Global (string pc (- e1 pc)) (string (inc e1) (- e2 e1 1)))) memo))
          0x93 (let [name (peek stack) module (peek (pop stack))]             ; STACK_GLOBAL
                 (recur pc (conj (pop (pop stack)) (->Global module name)) memo))
          0x71 (recur (inc pc) stack (assoc memo (u8 pc) top))                ; BINPUT
          0x72 (recur (+ pc 4) stack (assoc memo (u32 b pc) top))             ; LONG_BINPUT
          0x94 (recur pc stack (assoc memo (count memo) top))                 ; MEMOIZE
          0x68 (recur (inc pc) (conj stack (memo (u8 pc))) memo)             ; BINGET
          0x6a (recur (+ pc 4) (conj stack (memo (u32 b pc))) memo)          ; LONG_BINGET
          0x74 (let [[below items] (pop-mark stack)]                          ; TUPLE
                 (recur pc (conj below (vec items)) memo))
          0x85 (recur pc (conj pop1 [top]) memo)                              ; TUPLE1
          0x86 (let [s (count stack)]                                         ; TUPLE2
                 (recur pc (conj (subvec stack 0 (- s 2)) (subvec stack (- s 2))) memo))
          0x87 (let [s (count stack)]                                         ; TUPLE3
                 (recur pc (conj (subvec stack 0 (- s 3)) (subvec stack (- s 3))) memo))
          0x6c (let [[below items] (pop-mark stack)]                          ; LIST
                 (recur pc (conj below (vec items)) memo))
          0x64 (let [[below items] (pop-mark stack)]                          ; DICT
                 (recur pc (conj below (set-items {} items)) memo))
          0x61 (let [lst (peek pop1)]                                         ; APPEND
                 (recur pc (conj (pop pop1) (conj lst top)) memo))
          0x65 (let [[below items] (pop-mark stack)                           ; APPENDS
                     lst (peek below)]
                 (recur pc (conj (pop below) (into lst items)) memo))
          0x73 (let [v top k (peek pop1) d (peek (pop pop1))]                 ; SETITEM
                 (recur pc (conj (pop (pop pop1)) (assoc d k v)) memo))
          0x75 (let [[below items] (pop-mark stack)                           ; SETITEMS
                     d (peek below)]
                 (recur pc (conj (pop below) (set-items d items)) memo))
          0x52 (let [args top f (peek pop1)]                                  ; REDUCE
                 (recur pc (conj (pop pop1) (call-global load-storage f args)) memo))
          0x81 (let [args top cls (peek pop1)]                                ; NEWOBJ
                 (recur pc (conj (pop pop1) (call-global load-storage cls args)) memo))
          0x62 (recur pc pop1 memo)                                           ; BUILD: drop the state
          0x51 (recur pc (conj pop1 (persistent-load top)) memo)              ; BINPERSID
          0x30 (recur pc pop1 memo)                                           ; POP
          0x31 (let [[below _] (pop-mark stack)] (recur pc below memo))       ; POP_MARK
          0x32 (recur pc (conj stack top) memo)                               ; DUP
          (throw (ex-info (format "unsupported pickle opcode 0x%02x at %d" op (dec pc)) {:op op :pc (dec pc)})))))))

(declare load*)

(defn load
  "Load a torch.save'd object from a .pt file. With {:lazy true} tensors are
  read only when their :data (a delay) is forced, so picking a few tensors
  out of a large checkpoint reads only those."
  ([path] (load path {}))
  ([path {:keys [lazy]}]
   (binding [*lazy* (boolean lazy)]
     (load* path))))

(defn- load*
  [path]
  (let [path (str path)
        entries (zip-entries path)
        pkl (or (first (filter #(str/ends-with? % "data.pkl") (keys entries)))
                (throw (ex-info (str path " is not a torch zip checkpoint (no data.pkl)") {:path path})))
        prefix (subs pkl 0 (- (count pkl) (count "data.pkl")))
        {:keys [offset size method]} (entries pkl)
        _ (when-not (zero? method) (throw (ex-info "compressed data.pkl is not supported" {})))
        cache (atom {})
        load-storage (fn [storage]
                       (or (@cache (:key storage))
                           (let [data (read-storage-data path entries prefix storage)]
                             (swap! cache assoc (:key storage) data)
                             data)))]
    (unpickle (io/read-bytes path offset size) load-storage)))

(defn tensor? [x] (and (map? x) (contains? x :dtype) (contains? x :shape) (contains? x :data)))

(defn realize
  "A tensor with its data read (a no-op for eagerly loaded tensors)."
  [t]
  (if (delay? (:data t)) (assoc t :data @(:data t)) t))
