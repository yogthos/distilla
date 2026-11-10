(ns distilla.safetensors
  "Reader and writer for the safetensors format: an 8-byte little-endian
  header length, a JSON header naming every tensor's dtype, shape and byte
  range, then the raw data.

  Reads are lazy and ranged: `open` reads only the header, and `read-rows`
  reads one slice of the first dimension, which is how the embedding store
  pulls one collection's documents out of a multi-gigabyte shard."
  (:require [clojure.data.json :as json]
            [distilla.io :as io]
            [distilla.tensor :as t]))

(def dtype-sizes {"F64" 8 "F32" 4 "F16" 2 "BF16" 2 "I64" 8 "I32" 4 "I16" 2 "I8" 1 "U8" 1 "BOOL" 1})

(defn- le-u64 [^bytes b]
  (loop [i 7 v 0]
    (if (neg? i) v (recur (dec i) (bit-or (bit-shift-left v 8) (bit-and (aget b i) 0xff))))))

(defn open
  "Read a safetensors header: {:path :data-start :tensors {name {:dtype :shape :begin :end}} :metadata}."
  [path]
  (let [path (str path)
        n (le-u64 (io/read-bytes path 0 8))
        header (json/read-str (String. ^bytes (io/read-bytes path 8 n) "UTF-8"))]
    {:path path
     :data-start (+ 8 n)
     :metadata (get header "__metadata__")
     :tensors (into {}
                    (for [[k v] header :when (not= k "__metadata__")]
                      [k {:dtype (get v "dtype")
                          :shape (vec (get v "shape"))
                          :begin (first (get v "data_offsets"))
                          :end (second (get v "data_offsets"))}]))}))

(defn tensor-names [st] (sort (keys (:tensors st))))

(defn info [st name]
  (or (get-in st [:tensors name])
      (throw (ex-info (str "no tensor " (pr-str name) " in " (:path st)) {:name name :path (:path st)}))))

(defn shape [st name] (:shape (info st name)))

(defn- numel [shape] (reduce * 1 shape))

(defn- decode
  "Read n elements of dtype at an absolute file offset into a jolt array:
  floats for the floating dtypes (float16/bfloat16 widened), longs for the
  integer dtypes, booleans as bytes."
  [path offset dtype n]
  (case dtype
    "F32" (io/read-array path offset :float n)
    "F16" (t/halves->floats (io/read-array path offset :uint16 n))
    "BF16" (t/bf16s->floats (io/read-array path offset :uint16 n))
    "F64" (let [d (io/read-array path offset :double n)] (float-array (alength d) (seq d)))
    "I64" (io/read-array path offset :int64 n)
    "I32" (long-array (io/read-array path offset :int32 n))
    "I16" (long-array (io/read-array path offset :int16 n))
    ("U8" "I8" "BOOL") (io/read-array path offset :uint8 n)
    (throw (ex-info (str "unsupported dtype " dtype) {:dtype dtype}))))

(defn read-tensor
  "A whole tensor: {:dtype :shape :data} with :data a float array (floating
  dtypes) or a long array (integer dtypes)."
  [st name]
  (let [{:keys [dtype shape begin]} (info st name)]
    {:dtype dtype :shape shape
     :data (decode (:path st) (+ (:data-start st) begin) dtype (numel shape))}))

(defn read-rows
  "Rows [start, end) of the first dimension of a tensor, same shape as read-tensor."
  [st name start end]
  (let [{:keys [dtype shape begin]} (info st name)
        row-numel (numel (rest shape))
        row-bytes (* row-numel (dtype-sizes dtype))
        n (- end start)]
    (when (or (neg? start) (> end (first shape)) (neg? n))
      (throw (ex-info "row range out of bounds" {:start start :end end :rows (first shape)})))
    {:dtype dtype :shape (into [n] (rest shape))
     :data (decode (:path st) (+ (:data-start st) begin (* start row-bytes)) dtype (* n row-numel))}))

(defn read-matrix
  "A 2-D floating tensor (or a row range of it) as a distilla.tensor matrix."
  ([st name]
   (let [{:keys [shape data]} (read-tensor st name)]
     (t/matrix (first shape) (if (= 2 (count shape)) (second shape) 1) data)))
  ([st name start end]
   (let [{:keys [shape data]} (read-rows st name start end)
         cols (if (= 2 (count shape)) (second shape) (numel (rest shape)))]
     (t/matrix (first shape) cols data))))

(defn load-file
  "Every tensor of a file: {name {:dtype :shape :data}}."
  [path]
  (let [st (open path)]
    (into {} (for [name (tensor-names st)] [name (read-tensor st name)]))))

;; --- writing -----------------------------------------------------------------

(defn- encode
  "Turn tensor data into [io-type array] for the target dtype."
  [dtype data]
  (case dtype
    "F32" [:float (if (instance? (class (float-array 0)) data) data (float-array (map float data)))]
    "F16" [:uint16 (t/floats->halves (if (instance? (class (float-array 0)) data) data (float-array (map float data))))]
    "BF16" [:uint16 (t/floats->bf16s (if (instance? (class (float-array 0)) data) data (float-array (map float data))))]
    "I64" [:int64 (long-array data)]
    "I32" [:int32 (int-array data)]
    "U8" [:uint8 (byte-array (map unchecked-byte data))]
    "BOOL" [:uint8 (byte-array (map #(if (and % (not= 0 %)) 1 0) data))]
    (throw (ex-info (str "unsupported dtype " dtype) {:dtype dtype}))))

(defn save-file
  "Write tensors {name {:dtype :shape :data}} to path. :data is a float array
  or a sequence of numbers; it is converted to :dtype. Names are written in
  sorted order, as the Python implementation does."
  ([tensors path] (save-file tensors path nil))
  ([tensors path metadata]
   (let [names (sort (keys tensors))
         encoded (into {} (for [n names] [n (encode (get-in tensors [n :dtype]) (get-in tensors [n :data]))]))
         sizes (into {} (for [n names]
                          (let [{:keys [dtype shape]} (tensors n)
                                bytes (* (numel shape) (dtype-sizes dtype))
                                [_ arr] (encoded n)]
                            (when-not (= (alength arr) (numel shape))
                              (throw (ex-info (str "tensor " n " has " (alength arr) " elements, shape says "
                                                   (numel shape))
                                              {:name n :shape shape})))
                            [n bytes])))
         offsets (loop [ns names off 0 acc {}]
                   (if-let [n (first ns)]
                     (recur (rest ns) (+ off (sizes n)) (assoc acc n [off (+ off (sizes n))]))
                     acc))
         header (cond-> (into {} (for [n names]
                                   [n {"dtype" (get-in tensors [n :dtype])
                                       "shape" (vec (get-in tensors [n :shape]))
                                       "data_offsets" (offsets n)}]))
                  metadata (assoc "__metadata__" metadata))
         text (json/write-str header)
         raw (.getBytes ^String text "UTF-8")
         pad (mod (- 8 (mod (alength raw) 8)) 8)
         header-bytes (.getBytes ^String (str text (apply str (repeat pad " "))) "UTF-8")
         n (alength header-bytes)
         len-bytes (byte-array (for [i (range 8)] (unchecked-byte (bit-and (bit-shift-right n (* 8 i)) 0xff))))]
     (io/with-file [f path "wb"]
       (io/write-chunk! f :uint8 len-bytes)
       (io/write-chunk! f :uint8 header-bytes)
       (doseq [name names]
         (let [[type arr] (encoded name)]
           (io/write-chunk! f type arr))))
     path)))
