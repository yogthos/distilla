(ns distilla.io
  "Bulk binary file I/O through C stdio: typed reads at an offset, straight
  into foreign memory or into a jolt array, and streamed writes of arrays.
  The host is assumed little-endian, as the safetensors and torch formats are."
  (:require [jolt.ffi :as ffi]))

(ffi/defcfn c-fopen "fopen" [:string :string] :pointer)
(ffi/defcfn c-fclose "fclose" [:pointer] :int)
(ffi/defcfn c-fread "fread" [:pointer :size_t :size_t :pointer] :size_t :blocking)
(ffi/defcfn c-fwrite "fwrite" [:pointer :size_t :size_t :pointer] :size_t :blocking)
(ffi/defcfn c-fseeko "fseeko" [:pointer :int64 :int] :int)
(ffi/defcfn c-ftello "ftello" [:pointer] :int64)

(def ^:private seek-set 0)

(def type-sizes {:float 4 :double 8 :uint16 2 :int16 2 :int32 4 :int64 8 :uint8 1})

(defn open-file
  "fopen, throwing on failure."
  [path mode]
  (let [f (c-fopen (str path) mode)]
    (when (ffi/null? f)
      (throw (ex-info (str "cannot open " path ": " (ffi/errno-message)) {:path (str path) :mode mode})))
    f))

(defmacro with-file [[sym path mode] & body]
  `(let [~sym (open-file ~path ~mode)]
     (try ~@body (finally (c-fclose ~sym)))))

(defn read-into!
  "Read nbytes at offset of an open file into the foreign pointer dst."
  [f offset dst nbytes]
  (when (pos? nbytes)
    (when-not (zero? (c-fseeko f (long offset) seek-set))
      (throw (ex-info "seek failed" {:offset offset})))
    (let [got (c-fread dst 1 nbytes f)]
      (when-not (= got nbytes)
        (throw (ex-info "short read" {:offset offset :wanted nbytes :got got})))))
  dst)

(defn read-file-into!
  "Read nbytes at offset of the file at path into the foreign pointer dst."
  [path offset dst nbytes]
  (with-file [f path "rb"]
    (read-into! f offset dst nbytes)))

(defn read-array
  "Read n elements of a scalar type (:float :uint16 :int32 :int64 :uint8 ...)
  at offset: a float/short/int/long/byte array."
  [path offset type n]
  (if (zero? n)
    (case type
      :float (float-array 0) :double (double-array 0) (:uint16 :int16) (short-array 0)
      :int32 (int-array 0) :int64 (long-array 0) :uint8 (byte-array 0))
    (with-open [arena (ffi/confined-arena)]
      (let [nbytes (* n (type-sizes type))
            p (ffi/alloc arena nbytes)]
        (read-file-into! path offset p nbytes)
        (if (= type :uint8)
          (ffi/read-array p nbytes)
          (ffi/read-array p type n))))))

(defn read-bytes
  "n bytes at offset as a byte array."
  [path offset n]
  (read-array path offset :uint8 n))

(defn write-chunk!
  "Write an array (byte, short, int, long, float or double) to an open file."
  [f type arr]
  (let [n (alength arr)]
    (when (pos? n)
      (with-open [arena (ffi/confined-arena)]
        (let [nbytes (* n (type-sizes type))
              p (ffi/alloc arena nbytes)]
          (if (= type :uint8) (ffi/write-array p arr) (ffi/write-array p type arr))
          (when-not (= nbytes (c-fwrite p 1 nbytes f))
            (throw (ex-info "short write" {:bytes nbytes}))))))))

(defn file-size [path]
  (with-file [f path "rb"]
    (c-fseeko f 0 2)
    (c-ftello f)))
