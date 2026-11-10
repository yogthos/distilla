(ns distilla.porter
  "The Porter stemmer in nltk's default NLTK_EXTENSIONS mode, which both
  rouge_score (ROUGE with use_stemmer=True) and nltk's METEOR use. Ported
  rule for rule from nltk.stem.porter so that stems match exactly."
  (:require [clojure.string :as str]))

(def ^:private pool
  (into {} (for [[k vs] {"sky" ["sky" "skies"] "die" ["dying"] "lie" ["lying"] "tie" ["tying"]
                         "news" ["news"] "inning" ["innings" "inning"] "outing" ["outings" "outing"]
                         "canning" ["cannings" "canning"] "howe" ["howe"] "proceed" ["proceed"]
                         "exceed" ["exceed"] "succeed" ["succeed"]}
                 v vs]
             [v k])))

(def ^:private vowels #{\a \e \i \o \u})

(defn- consonant? [^String word i]
  (let [c (.charAt word i)]
    (cond
      (vowels c) false
      (= c \y) (loop [i i negate false]
                 (if (and (pos? i) (= \y (.charAt word i)))
                   (recur (dec i) (not negate))
                   (not= (not (vowels (.charAt word i))) negate)))
      :else true)))

(defn- consonant-flags [^String word]
  (loop [i 0 flags []]
    (if (= i (count word))
      flags
      (let [c (.charAt word i)]
        (recur (inc i)
               (conj flags (cond (vowels c) false
                                 (= c \y) (if (zero? i) true (not (nth flags (dec i))))
                                 :else true)))))))

(defn- measure [stem]
  (let [cv (apply str (map #(if % "c" "v") (consonant-flags stem)))]
    (count (re-seq #"vc" cv))))

(defn- positive-measure? [stem] (pos? (measure stem)))

(defn- contains-vowel? [stem] (not (every? true? (consonant-flags stem))))

(defn- ends-double-consonant? [^String word]
  (let [n (count word)]
    (and (>= n 2) (= (.charAt word (dec n)) (.charAt word (- n 2))) (consonant? word (dec n)))))

(defn- ends-cvc? [^String word]
  (let [n (count word)]
    (or (and (>= n 3)
             (consonant? word (- n 3))
             (not (consonant? word (- n 2)))
             (consonant? word (dec n))
             (not (#{\w \x \y} (.charAt word (dec n)))))
        (and (= n 2) (not (consonant? word 0)) (consonant? word 1)))))

(defn- strip-suffix [word suffix] (subs word 0 (- (count word) (count suffix))))

(defn- apply-rules
  "Apply the first rule whose suffix the word ends with. A rule is
  [suffix replacement condition]; the suffix \"*d\" means a double consonant."
  [word rules]
  (loop [rules rules]
    (if-let [[suffix replacement condition] (first rules)]
      (cond
        (and (= suffix "*d") (ends-double-consonant? word))
        (let [stem (subs word 0 (- (count word) 2))]
          (if (or (nil? condition) (condition stem)) (str stem replacement) word))

        (str/ends-with? word suffix)
        (let [stem (strip-suffix word suffix)]
          (if (or (nil? condition) (condition stem)) (str stem replacement) word))

        :else (recur (rest rules)))
      word)))

(defn- step1a [word]
  (if (and (str/ends-with? word "ies") (= 4 (count word)))
    (str (strip-suffix word "ies") "ie")
    (apply-rules word [["sses" "ss" nil] ["ies" "i" nil] ["ss" "ss" nil] ["s" "" nil]])))

(defn- step1b [word]
  (cond
    (str/ends-with? word "ied")
    (str (strip-suffix word "ied") (if (= 4 (count word)) "ie" "i"))

    (str/ends-with? word "eed")
    (let [stem (strip-suffix word "eed")]
      (if (pos? (measure stem)) (str stem "ee") word))

    :else
    (let [intermediate (some (fn [suffix]
                               (when (str/ends-with? word suffix)
                                 (let [stem (strip-suffix word suffix)]
                                   (when (contains-vowel? stem) stem))))
                             ["ed" "ing"])]
      (if-not intermediate
        word
        (let [last-char (str (last intermediate))]
          (apply-rules intermediate
                       [["at" "ate" nil]
                        ["bl" "ble" nil]
                        ["iz" "ize" nil]
                        ["*d" last-char (fn [_] (not (#{"l" "s" "z"} last-char)))]
                        ["" "e" (fn [stem] (and (= 1 (measure stem)) (ends-cvc? stem)))]]))))))

(defn- step1c [word]
  (apply-rules word [["y" "i" (fn [stem] (and (> (count stem) 1) (consonant? stem (dec (count stem)))))]]))

(defn- step2 [word]
  (if (and (str/ends-with? word "alli") (positive-measure? (strip-suffix word "alli")))
    (recur (str (strip-suffix word "alli") "al"))
    (apply-rules word
                 [["ational" "ate" positive-measure?]
                  ["tional" "tion" positive-measure?]
                  ["enci" "ence" positive-measure?]
                  ["anci" "ance" positive-measure?]
                  ["izer" "ize" positive-measure?]
                  ["bli" "ble" positive-measure?]
                  ["alli" "al" positive-measure?]
                  ["entli" "ent" positive-measure?]
                  ["eli" "e" positive-measure?]
                  ["ousli" "ous" positive-measure?]
                  ["ization" "ize" positive-measure?]
                  ["ation" "ate" positive-measure?]
                  ["ator" "ate" positive-measure?]
                  ["alism" "al" positive-measure?]
                  ["iveness" "ive" positive-measure?]
                  ["fulness" "ful" positive-measure?]
                  ["ousness" "ous" positive-measure?]
                  ["aliti" "al" positive-measure?]
                  ["iviti" "ive" positive-measure?]
                  ["biliti" "ble" positive-measure?]
                  ["fulli" "ful" positive-measure?]
                  ["logi" "log" (fn [_] (positive-measure? (subs word 0 (- (count word) 3))))]])))

(defn- step3 [word]
  (apply-rules word [["icate" "ic" positive-measure?]
                     ["ative" "" positive-measure?]
                     ["alize" "al" positive-measure?]
                     ["iciti" "ic" positive-measure?]
                     ["ical" "ic" positive-measure?]
                     ["ful" "" positive-measure?]
                     ["ness" "" positive-measure?]]))

(defn- step4 [word]
  (let [gt1 (fn [stem] (> (measure stem) 1))]
    (apply-rules word [["al" "" gt1] ["ance" "" gt1] ["ence" "" gt1] ["er" "" gt1] ["ic" "" gt1]
                       ["able" "" gt1] ["ible" "" gt1] ["ant" "" gt1] ["ement" "" gt1] ["ment" "" gt1]
                       ["ent" "" gt1]
                       ["ion" "" (fn [stem] (and (> (measure stem) 1) (#{\s \t} (last stem))))]
                       ["ou" "" gt1] ["ism" "" gt1] ["ate" "" gt1] ["iti" "" gt1] ["ous" "" gt1]
                       ["ive" "" gt1] ["ize" "" gt1]])))

(defn- step5a [word]
  (if (str/ends-with? word "e")
    (let [stem (strip-suffix word "e")
          m (measure stem)]
      (if (or (> m 1) (and (= m 1) (not (ends-cvc? stem)))) stem word))
    word))

(defn- step5b [word]
  (apply-rules word [["ll" "l" (fn [_] (> (measure (subs word 0 (dec (count word)))) 1))]]))

(defn stem
  "The Porter stem of a word (lower-cased first)."
  [word]
  (let [lowered (str/lower-case word)]
    (cond
      (contains? pool lowered) (pool lowered)
      (<= (count word) 2) lowered
      :else (-> lowered step1a step1b step1c step2 step3 step4 step5a step5b))))
