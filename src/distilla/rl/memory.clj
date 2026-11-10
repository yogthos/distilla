(ns distilla.rl.memory
  "Trajectory memory documents and action parsing shared by the TextWorld
  environments (SkyRL-Memorilla skyrl_gym/envs/textworld/memory.py)."
  (:require [clojure.string :as str]))

(def action-pattern #"(?i)\[ACTION:\s*([^\]]+)\]")
(def allowed-action-chars
  (set (str "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" " \t\n\r\u000b\f" "-_.,!?'")))
(def fallback-prefixes ["Action:" "I will" "I'll" "Let me" "I would" "I should"])
(def max-action-chars 200)

(defn- strip-chars [s chars]
  (let [cs (set chars)]
    (loop [s s]
      (cond (and (seq s) (cs (first s))) (recur (subs s 1))
            (and (seq s) (cs (last s))) (recur (subs s 0 (dec (count s))))
            :else s))))

(defn parse-action
  "The game command in a model response: the text of the first [ACTION: ...]
  tag, otherwise the first line after any </think> block minus a leading
  phrase such as \"I will\". Characters the parser can't take are removed and
  the command is capped at 200 characters. Empty when nothing usable."
  [response]
  (let [response (or response "")]
    (if (empty? response)
      ""
      (let [command (if-let [[_ c] (re-find action-pattern response)]
                      (str/trim c)
                      (let [c (str/trim (-> response (str/replace "<think>" "") (str/replace "</think>" "")))
                            c (if (str/includes? response "</think>")
                                (let [after (str/trim (second (str/split response #"</think>" 2)))]
                                  (if (seq after) after c))
                                c)
                            c (reduce (fn [c prefix]
                                        (if (str/starts-with? (str/lower-case c) (str/lower-case prefix))
                                          (str/trim (subs c (count prefix)))
                                          c))
                                      c fallback-prefixes)
                            c (if (str/includes? c "\n") (str/trim (first (str/split c #"\n" 2))) c)]
                        (strip-chars c "\"'.")))
            command (apply str (filter allowed-action-chars command))
            command (str/trim command)]
        (subs command 0 (min max-action-chars (count command)))))))

(defn trajectory-memory
  "Groups completed turns into documents: every turns-per-document turns make
  one, only the most recent max-documents are kept, and observations are cut
  to max-observation-chars (0 keeps them whole)."
  [turns-per-document max-documents max-observation-chars]
  (atom {:turns-per-document turns-per-document :max-documents max-documents
         :max-observation-chars max-observation-chars :documents [] :segment []}))

(defn- python-num
  "A float as Python's str() prints it (1.0, 0.0, 2.5)."
  [x]
  (let [d (double x)]
    (if (and (== d (Math/rint d)) (< (Math/abs d) 1e16))
      (str (long d) ".0")
      (str d))))

(defn- render [segment]
  (let [total (reduce + 0.0 (map :reward segment))
        change (- (:score (peek segment)) (:score (first segment)))
        lines (concat [(str "Turns " (:turn (first segment)) "-" (:turn (peek segment))
                            " | Reward: " (python-num total) " | Score Change: " change)
                       ""]
                      (mapcat (fn [{:keys [turn observation action reward]}]
                                (concat [(str "Turn " turn ":")]
                                        (when (seq observation) [(str "Obs: " observation)])
                                        [(str "Act: " action (when-not (zero? reward) (str " -> Reward: " (python-num reward))))
                                         ""]))
                              segment))]
    (str/trim (str/join "\n" lines))))

(defn flush!
  "Close the current document even if it holds fewer turns."
  [mem]
  (swap! mem (fn [{:keys [segment max-documents documents] :as m}]
               (if (empty? segment)
                 m
                 (let [docs (conj documents (render segment))]
                   (assoc m :segment [] :documents (vec (take-last max-documents docs))))))))

(defn add-turn!
  "Record a completed turn; the document closes once it holds enough turns."
  [mem {:keys [turn observation action reward score]}]
  (let [{:keys [max-observation-chars turns-per-document]} @mem
        observation (if (pos? max-observation-chars)
                      (subs observation 0 (min max-observation-chars (count observation)))
                      observation)]
    (swap! mem update :segment conj {:turn turn :observation observation :action action
                                     :reward (double reward) :score (long score)})
    (when (>= (count (:segment @mem)) turns-per-document)
      (flush! mem))))

(defn documents [mem] (:documents @mem))
