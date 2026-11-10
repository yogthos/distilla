(ns distilla.openai
  "A minimal OpenAI-compatible client over curl, for the data-generation
  pipelines: the Responses API (what the Python PersonalizationV4 pipeline
  calls) or chat/completions (any compatible server, e.g. llama-server).

  A client is a map:
    :api               :responses (default) or :chat
    :base-url          default $OPENAI_BASE_URL or https://api.openai.com/v1
    :api-key           default $OPENAI_API_KEY
    :max-retries       attempts per request (3), exponential backoff from 2 s
    :max-workers       concurrent requests (8)
    :complete-fn       optional fn [prompt model reasoning-effort] -> text,
                       which replaces HTTP entirely (tests, other backends)"
  (:require [clojure.data.json :as json]
            [clojure.string :as str]
            [jolt.process :as p]))

(def retry-delay-ms 2000)

(defn client [opts]
  (merge {:api :responses
          :base-url (or (not-empty (System/getenv "OPENAI_BASE_URL")) "https://api.openai.com/v1")
          :api-key (System/getenv "OPENAI_API_KEY")
          :max-retries 3
          :max-workers 8}
         opts))

(defn- body [{:keys [api]} prompt model reasoning-effort]
  (json/write-str
   (if (= api :chat)
     (cond-> {:model model :messages [{:role "user" :content prompt}]}
       (and reasoning-effort (not= reasoning-effort "none")) (assoc :reasoning_effort reasoning-effort))
     {:model model :input [{:role "user" :content prompt}] :reasoning {:effort reasoning-effort}})))

(defn response-text
  "The output text of a response body: the Responses API's output_text parts,
  or the first chat completion's message."
  [api parsed]
  (if (= api :chat)
    (get-in parsed ["choices" 0 "message" "content"])
    (or (get parsed "output_text")
        (str/join (for [item (get parsed "output")
                        part (get item "content")
                        :when (= "output_text" (get part "type"))]
                    (get part "text"))))))

(defn- start [{:keys [api base-url api-key] :as c} prompt model effort]
  (p/process {:in (body c prompt model effort) :out :string :err :string}
             "curl" "-sS" "-X" "POST" (str base-url (if (= api :chat) "/chat/completions" "/responses"))
             "-H" "Content-Type: application/json"
             "-H" (str "Authorization: Bearer " api-key)
             "--data-binary" "@-"))

(defn- finish [{:keys [api]} proc]
  (let [{:keys [exit out err]} @proc]
    (if-not (zero? exit)
      {:error (str "curl: " (str/trim (str err)))}
      (let [parsed (try (json/read-str out) (catch Exception _ nil))
            text (when parsed (response-text api parsed))]
        (cond
          (nil? parsed) {:error (str "unreadable response: " (subs out 0 (min 200 (count out))))}
          (get parsed "error") {:error (str (get-in parsed ["error" "message"] (get parsed "error")))}
          (str/blank? text) {:error "empty response"}
          :else {:text text})))))

(defn complete-all
  "Complete every request [prompt model reasoning-effort], max-workers at a
  time, each retried up to max-retries times. Returns one result per request,
  {:text} or {:error}, in order."
  [{:keys [complete-fn max-retries max-workers] :as c} requests]
  (let [requests (vec requests)
        attempt (fn [[prompt model effort]]
                  (if complete-fn
                    (try (let [t (complete-fn prompt model effort)]
                           (if (str/blank? t) {:error "empty response"} {:text t}))
                         (catch Exception e {:error (ex-message e)}))
                    (finish c (start c prompt model effort))))
        run-wave (fn [indices tries]
                   (if complete-fn
                     (into {} (for [i indices] [i (attempt (nth requests i))]))
                     (into {} (mapcat (fn [part]
                                        (let [procs (mapv (fn [i] [i (apply start c (nth requests i))]) part)]
                                          (for [[i proc] procs] [i (finish c proc)])))
                                      (partition-all max-workers indices)))))]
    (loop [pending (vec (range (count requests))) tries 0 results {}]
      (let [wave (run-wave pending tries)
            results (merge results wave)
            failed (filterv #(:error (wave %)) pending)]
        (if (or (empty? failed) (>= (inc tries) max-retries))
          (mapv results (range (count requests)))
          (do (Thread/sleep (long (* retry-delay-ms (Math/pow 2 tries))))
              (recur failed (inc tries) results)))))))

(defn complete
  "One request; the text, or throws after max-retries failures."
  [c prompt model reasoning-effort]
  (let [[{:keys [text error]}] (complete-all c [[prompt model reasoning-effort]])]
    (or text (throw (ex-info (str "No response from " model " after " (:max-retries c) " attempts: " error) {})))))
