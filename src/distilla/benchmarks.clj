(ns distilla.benchmarks
  "Benchmark registry: evaluation protocol for the memory model and for the
  closed-book, RAG and full-context baselines."
  (:require [clojure.set :as set]))

(def choice "choice")
(def exact-match "exact_match")
(def polarity "polarity")
(def generation "generation")
(def scorings [choice exact-match polarity generation])

(def accuracy "accuracy")
(def accuracy-hard "accuracy_hard")
(def bleu1 "bleu1")
(def bleu4 "bleu4")
(def rouge1 "rouge1")
(def rougeL "rougeL")
(def f1 "f1")
(def meteor "meteor")
(def generation-metrics [bleu1 bleu4 rouge1 rougeL f1 meteor])
(def lamp-metrics [rouge1 rougeL f1 meteor bleu1 bleu4])
(def triviaqa-metrics [exact-match bleu1 bleu4 f1 rouge1 rougeL meteor])
(def metrics-by-scoring
  {choice [accuracy accuracy-hard]
   exact-match (into [exact-match] generation-metrics)
   polarity [accuracy]
   generation generation-metrics})

(def retrieval-context-length 32768)
(def yarn-original-max-position-embeddings 32768)

(def persona-prompt
  "You are a personalized AI assistant. Answer the question about the user based on your understanding of the user.")
(def trivia-prompt
  (str "You are a helpful assistant that answers trivia questions using a compressed memory of retrieved Wikipedia "
       "articles. Answer concisely with a short factual phrase or single word."))
(def helpful-prompt "You are a helpful assistant.")
(def fact-check-prompt
  (str "You are a fact-verification assistant. Decide whether the given claim is true or false based on your knowledge. "
       "Output exactly 'True' or 'False' with no other text."))
(def brief-answer-prompt
  (str "You are a question-answering assistant. Provide a direct, brief answer in 1-2 sentences. "
       "Do not include preamble, context, or explanation."))
(def personalized-style-prompt
  (str "You are a personalized AI assistant. Use your memory of the user's history to respond in their style and "
       "preferences."))

(def trivia-suffix
  (str "\n\nAnswer with the exact short factual phrase that answers the question. For example:\n"
       "- \"Mark Twain\"\n"
       "- \"1969\"\n"
       "- \"Mount Everest\"\n"
       "- \"the Pacific Ocean\"\n"
       "- \"carbon dioxide\"\n"
       "\nYour answer:"))
(def fact-check-suffix
  (str "\n\nOutput ONLY \"True\" or \"False\". Do not include explanations, rephrasings, or any other text. Examples:\n"
       "\nQ: Is the following claim true or false? Claim: Albert Einstein was born in Germany.\nA: True\n"
       "\nQ: Is the following claim true or false? Claim: The Eiffel Tower is in London.\nA: False\n"
       "\nQ: Is the following claim true or false? Claim: Water boils at 100 degrees Celsius at standard atmospheric "
       "pressure.\nA: True\n"
       "\nQ: Is the following claim true or false? Claim: Mars has more moons than Jupiter.\nA: False\n"
       "\nQ: Is the following claim true or false? Claim: William Shakespeare wrote Hamlet.\nA: True\n"
       "\nA:\n"))
(def polarity-suffix
  (str "\nAnswer the research question with one of \"Yes.\", \"No.\", or \"Maybe.\" followed by a brief justification, in "
       "the format:\n"
       "- \"Yes. UDR provides an objective measurement of VUR and appears as a predictive tool of success after "
       "endoscopic injection.\"\n"
       "- \"Yes. Serum TB level was independently associated with cardioembolic stroke.\"\n"
       "- \"No. The results of this study cast doubt on the suggested advantage of HBO in reducing patient mortality and "
       "morbidity.\"\n"
       "- \"No. Baseline pain intensity does not predict the outcome after an appropriate opioid titration.\"\n"
       "- \"Maybe. The body mass index is one of the prognostic factors of stage 2 and 3a gastric cancer but not for "
       "other stages.\"\n"
       "- \"Maybe. Limb-salvage surgery offers better gait efficiency than above-knee amputation, but does not improve "
       "perceived quality of life.\"\n"
       "\nYour answer:"))

(defn validate-metrics
  "Check that a scoring method is known and produces every requested metric."
  [scoring metrics]
  (when-not (some #{scoring} scorings)
    (throw (ex-info (str "Unknown scoring " (pr-str scoring) "; expected one of " (pr-str scorings) ".")
                    {:scoring scoring})))
  (let [unknown (set/difference (set metrics) (set (metrics-by-scoring scoring)))]
    (when (seq unknown)
      (throw (ex-info (str "Metrics " (pr-str (vec (sort unknown))) " are not produced by "
                           (pr-str scoring) " scoring.")
                      {:scoring scoring :metrics (vec (sort unknown))})))))

(defn baseline-protocol
  "Prompting and context settings for the text-only baselines."
  [system-prompt max-new-tokens full-context-length & {:keys [suffix yarn-factor] :or {suffix ""}}]
  {:system-prompt system-prompt
   :max-new-tokens max-new-tokens
   :full-context-length full-context-length
   :suffix suffix
   :yarn-factor yarn-factor})

(defn benchmark
  "One evaluation benchmark, validated. Keys: :name :config :split
  :system-prompt :max-length :max-new-tokens :scoring :metrics, optional
  :baseline and :suffix."
  [{:keys [name scoring metrics] :as spec}]
  (try
    (validate-metrics scoring metrics)
    (catch clojure.lang.ExceptionInfo e
      (throw (ex-info (str name ": " (ex-message e)) (ex-data e)))))
  (merge {:suffix "" :baseline nil} spec {:metrics (vec metrics)}))

(def benchmark-list
  [(benchmark {:name "pmv2" :config "pmv2" :split "test" :system-prompt persona-prompt
               :max-length 8192 :max-new-tokens 256 :scoring choice :metrics [accuracy]
               :baseline (baseline-protocol helpful-prompt 256 40960)})
   (benchmark {:name "pv4" :config "pv4" :split "test" :system-prompt persona-prompt
               :max-length 8192 :max-new-tokens 256 :scoring choice :metrics [accuracy accuracy-hard]
               :baseline (baseline-protocol helpful-prompt 256 65536 :yarn-factor 2.0)})
   (benchmark {:name "factkg" :config "factkg" :split "test" :system-prompt persona-prompt
               :max-length 2048 :max-new-tokens 8 :scoring choice :metrics [accuracy]
               :baseline (baseline-protocol fact-check-prompt 8 8192 :suffix fact-check-suffix)})
   (benchmark {:name "triviaqa" :config "triviaqa" :split "test" :system-prompt trivia-prompt
               :max-length 2048 :max-new-tokens 64 :scoring exact-match :metrics triviaqa-metrics
               :baseline (baseline-protocol helpful-prompt 64 131072 :suffix trivia-suffix :yarn-factor 4.0)
               :suffix trivia-suffix})
   (benchmark {:name "narrativeqa" :config "narrativeqa" :split "test" :system-prompt persona-prompt
               :max-length 8192 :max-new-tokens 32 :scoring generation :metrics generation-metrics
               :baseline (baseline-protocol brief-answer-prompt 64 131072 :yarn-factor 4.0)})
   (benchmark {:name "pubmedqa" :config "pubmedqa" :split "test" :system-prompt persona-prompt
               :max-length 8192 :max-new-tokens 64 :scoring polarity :metrics [accuracy]
               :baseline (baseline-protocol helpful-prompt 64 8192 :suffix polarity-suffix)})
   (benchmark {:name "lamp4" :config "lamp4" :split "test" :system-prompt persona-prompt
               :max-length 2048 :max-new-tokens 128 :scoring generation :metrics lamp-metrics
               :baseline (baseline-protocol personalized-style-prompt 128 32768)})
   (benchmark {:name "lamp7" :config "lamp7" :split "test" :system-prompt persona-prompt
               :max-length 1024 :max-new-tokens 128 :scoring generation :metrics lamp-metrics
               :baseline (baseline-protocol personalized-style-prompt 128 8192)})])

(def benchmarks
  "Registry keyed by name; `benchmark-names` keeps the registry order."
  (into {} (map (juxt :name identity)) benchmark-list))

(def benchmark-names (mapv :name benchmark-list))

(defn get-benchmarks
  "One benchmark by name, or every benchmark with \"all\", in registry order."
  [name]
  (cond
    (= name "all") benchmark-list
    (contains? benchmarks name) [(benchmarks name)]
    :else (throw (ex-info (str "Unknown benchmark " (pr-str name) "; choose from "
                               (pr-str (vec (sort (keys benchmarks)))) " or 'all'.")
                          {:name name}))))
