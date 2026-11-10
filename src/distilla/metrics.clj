(ns distilla.metrics
  "Scorers for every benchmark and the shared writer of predictions and
  metrics. Every metric is a fraction in [0, 1]; write-metrics reports them as
  percentages.

  BLEU (nltk corpus_bleu, method-1 smoothing), ROUGE (rouge_score with the
  Porter stemmer) and token F1 are ports that reproduce the Python numbers.
  METEOR follows nltk's single_meteor_score with its exact and stem matching
  stages; nltk's third stage, WordNet synonyms, is not available here, so
  METEOR can come out slightly lower than nltk's when a prediction uses a
  synonym of a reference word."
  (:require [babashka.fs :as fs]
            [clojure.data.json :as json]
            [clojure.set :as set]
            [clojure.string :as str]
            [distilla.benchmarks :as b]
            [distilla.dataset :as ds]
            [distilla.encoder :as encoder]
            [distilla.porter :as porter]
            [distilla.tensor :as t]
            [distilla.util :as util]))

(def chat-token-pattern #"<\|(?:im_end|im_start|endoftext|im_sep)\|>")
(def polarity-words #{"yes" "no" "maybe"})
(def list-markers #{\- \* \•})
(def quotes #{\" \'})
(def numbered-markers #{"1." "2." "3."})
(def max-prefix-strips 3)
(def predictions-file "predictions.jsonl")
(def metrics-file "metrics.json")
(def punctuation (set "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))

(defn clean-generation
  "Remove chat-template special tokens and surrounding whitespace."
  [text]
  (str/trim (str/replace text chat-token-pattern "")))

(defn- python-strip
  "str.strip(): whitespace including Unicode separators."
  [s]
  (str/trim s))

(defn- tokenize
  "Lower-case whitespace tokenisation used by the generation metrics."
  [text]
  (let [lowered (str/lower-case text)]
    (vec (remove str/blank? (str/split lowered #"\s+")))))

(defn- assert-same-length [predictions references]
  (when-not (= (count predictions) (count references))
    (throw (ex-info "predictions and references differ in length"
                    {:predictions (count predictions) :references (count references)}))))

;; --- BLEU ------------------------------------------------------------------------

(defn- ngrams [tokens n]
  (if (< (count tokens) n) [] (map vec (partition n 1 tokens))))

(defn- modified-precision
  "[numerator denominator] of nltk's modified n-gram precision."
  [references hypothesis n]
  (let [counts (frequencies (ngrams hypothesis n))
        max-counts (apply merge-with max {} (map #(frequencies (ngrams % n)) references))
        clipped (reduce + 0 (for [[g c] counts] (min c (get max-counts g 0))))]
    [clipped (max 1 (reduce + 0 (vals counts)))]))

(defn- closest-ref-length [references hyp-len]
  (->> references
       (map count)
       (sort-by (fn [len] [(Math/abs (long (- len hyp-len))) len]))
       first))

(defn- brevity-penalty [closest-ref-len hyp-len]
  (cond
    (> hyp-len closest-ref-len) 1.0
    (zero? hyp-len) 0.0
    :else (Math/exp (- 1.0 (/ (double closest-ref-len) hyp-len)))))

(defn corpus-bleu
  "nltk's corpus_bleu with method-1 smoothing (epsilon 0.1) for one weight vector."
  [list-of-references hypotheses weights]
  (let [max-n (count weights)
        stats (for [[refs hyp] (map vector list-of-references hypotheses)]
                {:p (mapv #(modified-precision refs hyp %) (range 1 (inc max-n)))
                 :hyp (count hyp)
                 :ref (closest-ref-length refs (count hyp))})
        numerators (vec (for [i (range max-n)] (reduce + 0 (map #(first (nth (:p %) i)) stats))))
        denominators (vec (for [i (range max-n)] (reduce + 0 (map #(second (nth (:p %) i)) stats))))
        hyp-lengths (reduce + 0 (map :hyp stats))
        ref-lengths (reduce + 0 (map :ref stats))
        bp (brevity-penalty ref-lengths hyp-lengths)]
    (if (zero? (first numerators))
      0.0
      (let [p-n (for [i (range max-n)]
                  (let [num (nth numerators i) den (nth denominators i)]
                    (if (zero? num) (/ 0.1 den) (/ (double num) den))))
            s (reduce + 0.0 (for [[w p] (map vector weights p-n) :when (pos? p)] (* w (Math/log p))))]
        (* bp (Math/exp s))))))

(defn bleu
  "Corpus BLEU-1 and BLEU-4 with method-1 smoothing."
  [predictions references]
  (let [hypotheses (mapv tokenize predictions)
        reference-lists (mapv (fn [r] [(tokenize r)]) references)]
    {b/bleu1 (corpus-bleu reference-lists hypotheses [1 0 0 0])
     b/bleu4 (corpus-bleu reference-lists hypotheses [0.25 0.25 0.25 0.25])}))

;; --- ROUGE -------------------------------------------------------------------------

(defn rouge-tokenize
  "rouge_score's tokenizer: lower-case, non-alphanumerics to spaces, Porter
  stems for tokens longer than three characters."
  [text]
  (let [text (str/replace (str/lower-case text) #"[^a-z0-9]+" " ")]
    (->> (str/split text #"\s+")
         (map #(if (> (count %) 3) (porter/stem %) %))
         (filter #(re-matches #"[a-z0-9]+" %))
         vec)))

(defn- fmeasure [precision recall]
  (if (pos? (+ precision recall)) (/ (* 2 precision recall) (+ precision recall)) 0.0))

(defn- rouge-n [target prediction n]
  (let [tg (frequencies (ngrams target n))
        pg (frequencies (ngrams prediction n))
        overlap (reduce + 0 (for [[g c] tg] (min c (get pg g 0))))
        precision (/ (double overlap) (max 1 (reduce + 0 (vals pg))))
        recall (/ (double overlap) (max 1 (reduce + 0 (vals tg))))]
    (fmeasure precision recall)))

(defn- lcs-length [a b]
  (let [a (vec a) b (vec b) cols (inc (count b))]
    (loop [i 0 prev (vec (repeat cols 0))]
      (if (= i (count a))
        (peek prev)
        (recur (inc i)
               (loop [j 1 row [0]]
                 (if (= j cols)
                   row
                   (recur (inc j)
                          (conj row (if (= (nth a i) (nth b (dec j)))
                                      (inc (nth prev (dec j)))
                                      (max (nth prev j) (peek row))))))))))))

(defn- rouge-l [target prediction]
  (if (or (empty? target) (empty? prediction))
    0.0
    (let [lcs (lcs-length target prediction)]
      (fmeasure (/ (double lcs) (count prediction)) (/ (double lcs) (count target))))))

(defn rouge
  "Mean ROUGE-1 and ROUGE-L F-measure with Porter stemming."
  [predictions references]
  (assert-same-length predictions references)
  (let [scores (for [[p r] (map vector predictions references)]
                 (let [target (rouge-tokenize r)
                       prediction (rouge-tokenize (python-strip p))]
                   [(rouge-n target prediction 1) (rouge-l target prediction)]))]
    {b/rouge1 (util/mean (map first scores))
     b/rougeL (util/mean (map second scores))}))

;; --- token F1 ------------------------------------------------------------------------

(defn token-f1
  "Mean word-overlap F1 on lower-cased whitespace tokens."
  [predictions references]
  (assert-same-length predictions references)
  (util/mean
   (for [[p r] (map vector predictions references)]
     (let [predicted (tokenize p) gold (tokenize r)]
       (if (or (empty? predicted) (empty? gold))
         0.0
         (let [pc (frequencies predicted) gc (frequencies gold)
               overlap (reduce + 0 (for [tok (set/intersection (set predicted) (set gold))]
                                     (min (pc tok) (gc tok))))]
           (if (zero? overlap)
             0.0
             (let [precision (/ (double overlap) (count predicted))
                   recall (/ (double overlap) (count gold))]
               (/ (* 2 precision recall) (+ precision recall))))))))))

;; --- METEOR ----------------------------------------------------------------------------

(defn- match-enums
  "nltk's _match_enums: [matches unmatched-hyp unmatched-ref], scanning the
  hypothesis from the end and taking the last free reference position."
  [hyp ref]
  (let [positions (reduce (fn [m [j [_ w]]] (update m w (fnil conj []) j)) {} (map-indexed vector ref))]
    (loop [i (dec (count hyp)) positions positions matches [] hyp-used #{} ref-used #{}]
      (if (neg? i)
        [matches
         (vec (keep-indexed (fn [k p] (when-not (hyp-used k) p)) hyp))
         (vec (keep-indexed (fn [k p] (when-not (ref-used k) p)) ref))]
        (let [[hi w] (nth hyp i)
              ps (get positions w)]
          (if (seq ps)
            (let [j (peek ps)]
              (recur (dec i) (assoc positions w (pop ps)) (conj matches [hi (first (nth ref j))])
                     (conj hyp-used i) (conj ref-used j)))
            (recur (dec i) positions matches hyp-used ref-used)))))))

(defn- count-chunks [matches]
  (loop [i 0 chunks 1]
    (if (< i (dec (count matches)))
      (let [[a b] (nth matches i) [c d] (nth matches (inc i))]
        (recur (inc i) (if (and (= c (inc a)) (= d (inc b))) chunks (inc chunks))))
      chunks)))

(defn single-meteor
  "METEOR of one tokenized hypothesis against one tokenized reference
  (alpha 0.9, beta 3, gamma 0.5), with exact and Porter-stem matching."
  [reference hypothesis]
  (let [hyp (vec (map-indexed vector (map str/lower-case hypothesis)))
        ref (vec (map-indexed vector (map str/lower-case reference)))
        [exact hyp' ref'] (match-enums hyp ref)
        stem-pairs (fn [xs] (mapv (fn [[i w]] [i (porter/stem w)]) xs))
        [stemmed _ _] (match-enums (stem-pairs hyp') (stem-pairs ref'))
        matches (vec (sort-by first (concat exact stemmed)))
        m (count matches)]
    (if (or (zero? m) (empty? hyp) (empty? ref))
      0.0
      (let [precision (/ (double m) (count hyp))
            recall (/ (double m) (count ref))
            fmean (/ (* precision recall) (+ (* 0.9 precision) (* 0.1 recall)))
            frag (/ (double (count-chunks matches)) m)
            penalty (* 0.5 (Math/pow frag 3.0))]
        (* (- 1.0 penalty) fmean)))))

(defn meteor
  "Mean single-reference METEOR on lower-cased whitespace tokens."
  [predictions references]
  (assert-same-length predictions references)
  (util/mean (for [[p r] (map vector predictions references)]
               (single-meteor (tokenize r) (tokenize p)))))

(defn ensure-meteor-resources
  "METEOR needs no external data here (nltk needs WordNet); kept so callers
  read like the Python evaluation."
  [] nil)

;; --- metric groups --------------------------------------------------------------------

(defn generation-metrics
  "BLEU-1/4, ROUGE-1/L, token F1 and METEOR, computing only what is requested."
  ([predictions references] (generation-metrics predictions references b/generation-metrics))
  ([predictions references names]
   (let [names (vec names)
         wanted (set names)
         scores (cond-> {}
                  (some wanted [b/bleu1 b/bleu4]) (merge (bleu predictions references))
                  (some wanted [b/rouge1 b/rougeL]) (merge (rouge predictions references))
                  (wanted b/f1) (assoc b/f1 (token-f1 predictions references))
                  (wanted b/meteor) (assoc b/meteor (meteor predictions references)))]
     (into (array-map) (for [n names] [n (get scores n)])))))

(defn normalize-answer
  "SQuAD normalisation: lower-case, drop punctuation and articles, collapse whitespace."
  [text]
  (if (empty? (str text))
    ""
    (let [text (apply str (remove punctuation (str/lower-case text)))
          text (str/replace text #"\b(a|an|the)\b" " ")]
      (str/join " " (remove str/blank? (str/split text #"\s+"))))))

(defn exact-match
  "Fraction of predictions equal to their reference after normalize-answer."
  [predictions references]
  (assert-same-length predictions references)
  (util/mean (for [[p r] (map vector predictions references)]
               (if (= (normalize-answer p) (normalize-answer r)) 1.0 0.0))))

(defn first-polarity
  "The leading yes/no/maybe of a text, skipping list markers, numbering and quotes."
  [text]
  (when-not (str/blank? (str text))
    (let [text (loop [text (str/trim text) k 0]
                 (cond
                   (or (= k max-prefix-strips) (empty? text)) text
                   (list-markers (first text)) (recur (str/triml (subs text 1)) (inc k))
                   (quotes (first text)) (recur (subs text 1) (inc k))
                   (numbered-markers (subs text 0 (min 2 (count text)))) (recur (str/triml (subs text 2)) (inc k))
                   :else text))]
      (when-not (str/blank? text)
        (let [head (-> (first (str/split (str/trim text) #"\s+"))
                       (str/replace #"[.,!?:;\"']+$" "")
                       str/lower-case)]
          (when (polarity-words head) head))))))

(defn polarity-accuracy
  "Fraction of predictions whose first word matches the reference's
  yes/no/maybe; references without one are skipped (0 when none has one)."
  [predictions references]
  (assert-same-length predictions references)
  (let [pairs (for [[p r] (map vector predictions references)
                    :let [gold (first-polarity r)]
                    :when gold]
                [(first-polarity p) gold])]
    (if (seq pairs)
      (/ (double (count (filter (fn [[p g]] (= p g)) pairs))) (count pairs))
      0.0)))

;; --- choice scoring ------------------------------------------------------------------------

(defn cosine-similarities
  "Cosine similarity between a vector and each row of a matrix (eps 1e-8)."
  [^floats vector {:keys [rows cols data]}]
  (let [vn (+ (t/norm vector) 1e-8)]
    (vec (for [i (range rows)]
           (let [r (float-array cols)]
             (System/arraycopy data (* i cols) r 0 cols)
             (/ (t/dot vector r) (* vn (+ (t/norm r) 1e-8))))))))

(defn choice-index
  "Position of the reference among a row's choices, ignoring surrounding whitespace."
  [choices reference]
  (let [stripped (mapv str/trim choices)
        idx (.indexOf ^java.util.List stripped (str/trim reference))]
    (when (neg? idx)
      (throw (ex-info (str "Reference answer " (pr-str reference) " is not one of the choices "
                           (pr-str (vec choices)) ".")
                      {:reference reference :choices choices})))
    idx))

(defn choice-predictions
  "For every prediction embedding (rows of a matrix), the index into its
  row's choices of the most cosine-similar non-empty choice.
  choice-embeddings holds per example a matrix of its non-empty choices."
  [prediction-embeddings choice-embeddings choices]
  (vec (for [i (range (:rows prediction-embeddings))]
         (let [texts (nth choices i)
               sims (cosine-similarities (t/row prediction-embeddings i) (nth choice-embeddings i))
               nonempty (vec (keep-indexed (fn [k s] (when-not (str/blank? s) k)) texts))
               scored (map vector nonempty sims)]
           (first (reduce (fn [[bi bs] [k s]] (if (> s bs) [k s] [bi bs]))
                          [0 Double/NEGATIVE_INFINITY] scored))))))

(defn choice-accuracy
  "Embedding nearest-neighbour accuracy over answer choices, plus
  accuracy_hard over the rows flagged in hard (when given)."
  [predictions references choices enc hard]
  (let [nonempty (mapv (fn [row] (vec (remove str/blank? row))) choices)
        vectors (encoder/embed-texts enc (concat (map str/trim predictions) (apply concat nonempty)))
        n (count predictions)
        choice-vectors (loop [rows nonempty offset n acc []]
                         (if-let [row (first rows)]
                           (recur (rest rows) (+ offset (count row))
                                  (conj acc (t/select-rows vectors (range offset (+ offset (count row))))))
                           acc))
        selected (choice-predictions (t/take-rows vectors n) choice-vectors choices)
        correct (mapv (fn [row reference index] (= (choice-index row reference) index))
                      choices references selected)
        accuracy (util/mean (map #(if % 1.0 0.0) correct))]
    (cond-> {b/accuracy accuracy}
      hard (assoc b/accuracy-hard
                  (let [flagged (keep-indexed (fn [i c] (when (nth hard i) c)) correct)]
                    (if (seq flagged) (util/mean (map #(if % 1.0 0.0) flagged)) 0.0))))))

(defn compute-metrics
  "Score predictions with one scoring method; the requested metrics as
  fractions, in the order of metrics. opts: :choices :hard :encoder."
  [scoring metrics predictions references & {:keys [choices hard encoder]}]
  (b/validate-metrics scoring metrics)
  (let [metrics (vec metrics)
        scores (cond
                 (= scoring b/choice)
                 (do (when (or (nil? choices) (nil? encoder))
                       (throw (ex-info "Choice scoring needs the answer choices and an embedding engine." {})))
                     (choice-accuracy predictions references choices encoder
                                      (when (some #{b/accuracy-hard} metrics) hard)))
                 (= scoring b/polarity) {b/accuracy (polarity-accuracy predictions references)}
                 (= scoring b/exact-match) {b/exact-match (exact-match predictions references)}
                 :else {})
        requested (filterv (set b/generation-metrics) metrics)
        scores (if (seq requested) (merge scores (generation-metrics predictions references requested)) scores)]
    (into (array-map) (for [m metrics] [m (get scores m)]))))

(defn check-scoring-inputs
  "Check, before generating, that a split can be scored with a benchmark's metrics."
  [benchmark dataset]
  (when (= (:scoring benchmark) b/choice)
    (when-not (ds/has-column? dataset "choices")
      (throw (ex-info (str (:name benchmark) ": choice scoring needs a \"choices\" column.") {})))
    (doseq [[choices answer] (map vector (ds/column dataset "choices") (ds/column dataset "answer"))]
      (choice-index choices (str answer))))
  (when (some #{b/meteor} (:metrics benchmark))
    (ensure-meteor-resources)))

(defn evaluation-records
  "Pair every evaluated row with its prediction."
  [dataset predictions]
  (let [optional (filterv #(ds/has-column? dataset %) ["choices" "hard"])]
    (mapv (fn [row prediction]
            (into (array-map "question" (get row "question")
                             "answer" (get row "answer")
                             "prediction" prediction)
                  (for [c optional] [c (get row c)])))
          (ds/rows dataset) predictions)))

(defn score-dataset
  "Score the predictions for one benchmark split."
  [benchmark dataset predictions encoder]
  (let [references (mapv #(str/trim (str %)) (ds/column dataset "answer"))
        choices (when (ds/has-column? dataset "choices") (ds/column dataset "choices"))
        hard (when (ds/has-column? dataset "hard") (ds/column dataset "hard"))]
    (compute-metrics (:scoring benchmark) (:metrics benchmark) predictions references
                     :choices choices :hard hard :encoder encoder)))

(defn write-predictions [output-dir records]
  (util/write-jsonl (fs/path output-dir predictions-file) records))

(defn write-metrics
  "Write metrics.json: percentages with two decimals and the number of examples."
  [output-dir scores num-examples]
  (let [summary (-> (into (array-map) (for [[k v] scores] [k (util/round2 (* 100.0 v))]))
                    (assoc "num_examples" num-examples))]
    (util/write-json (fs/path output-dir metrics-file) summary)
    summary))

(def ^:dynamic *score-dataset* score-dataset)

(defn score-predictions
  "Write every benchmark's predictions, then score them, write their metrics
  and print a summary line each. All predictions are on disk before scoring
  starts. jobs: [benchmark dataset predictions output-dir]. get-encoder is
  called once, and only when a benchmark uses choice scoring; closing what it
  returns is the caller's business."
  [jobs get-encoder]
  (doseq [[_ dataset predictions output-dir] jobs]
    (write-predictions output-dir (evaluation-records dataset predictions)))
  (let [enc (when (some (fn [[bm]] (= (:scoring bm) b/choice)) jobs) (get-encoder))]
    (into (array-map)
          (doall
           (for [[bm dataset predictions output-dir] jobs]
             (let [scores (*score-dataset* bm dataset predictions enc)
                   summary (write-metrics output-dir scores (count predictions))]
               (println (str (:name bm) ": " (json/write-str summary) "  ->  " output-dir))
               [(:name bm) summary]))))))
