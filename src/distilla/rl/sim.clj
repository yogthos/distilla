(ns distilla.rl.sim
  "A TextWorld game simulator: a rule-for-rule port of skyrl-gym's `FastTextWorldSimulator`
  (`skyrl_gym/envs/textworld/fast_sim.py`). It reads a TextWorld `.json` game spec and plays it
  in-process, producing the same observation text, rewards, scores and info maps as the Python.

      (def sim (load-game \"game.json\"))
      (let [[obs info] (reset! sim)] ...)
      (let [[obs reward done info] (step! sim \"go north\")] ...)

  The simulator is an atom holding a plain map. Where the Python iterates a dict, the port keeps
  an insertion-ordered vector of `[k v]` pairs (see `od-*`), so text built from that iteration
  comes out in the same order. The one place the Python iterates a set of strings, matching a name
  against the carried items, depends on the string hash seed when two items tie; the port (and the
  fixture generator, which patches the Python) uses the items sorted by id."
  (:refer-clojure :exclude [reset!])
  (:require [clojure.data.json :as json]
            [clojure.string :as str]))

;; ---------------------------------------------------------------------------
;; Insertion-ordered dicts: vectors of [k v], with Python dict semantics (assigning an existing key
;; keeps its position, deleting and re-adding moves it to the end).

(defn- od-index [od k]
  (loop [i 0]
    (cond (= i (count od)) nil
          (= k (first (nth od i))) i
          :else (recur (inc i)))))

(defn- od-get
  ([od k] (od-get od k nil))
  ([od k default] (if-let [i (od-index od k)] (second (nth od i)) default)))

(defn- od-has? [od k] (some? (od-index od k)))

(defn- od-put [od k v]
  (if-let [i (od-index od k)] (assoc od i [k v]) (conj od [k v])))

(defn- od-del [od k] (into [] (remove #(= k (first %))) od))

(defn- od-keys-where
  "Keys whose value is `v`, in insertion order."
  [od v]
  (keep (fn [[k x]] (when (= x v) k)) od))

;; ---------------------------------------------------------------------------
;; Small Python string helpers

(defn- title-case
  "Python's str.title(): a cased letter is upper-cased after an uncased character, lower-cased
  after a cased one."
  [s]
  (let [sb (StringBuilder.)]
    (loop [cs (seq s) prev-cased? false]
      (if-let [c (first cs)]
        (let [cased? (or (Character/isUpperCase c) (Character/isLowerCase c))]
          (.append sb (if cased?
                        (if prev-cased? (Character/toLowerCase c) (Character/toUpperCase c))
                        c))
          (recur (rest cs) cased?))
        (str sb)))))

(defn- re-split
  "Python's re.split: the pieces between matches, with each match's groups spliced in between."
  [re s]
  (let [m (re-matcher re s)]
    (loop [start 0 out []]
      (if (.find m)
        (recur (.end m)
               (into (conj out (subs s start (.start m)))
                     (map #(.group m %) (range 1 (inc (.groupCount m))))))
        (conj out (subs s start))))))

(defn- replace-first-literal
  "Python's str.replace(old, new, 1)."
  [s old new]
  (if-let [i (str/index-of s old)]
    (str (subs s 0 i) new (subs s (+ i (count old))))
    s))

(defn- strip-article [name-lower]
  (or (some (fn [article]
              (when (str/starts-with? name-lower article)
                (subs name-lower (count article))))
            ["the " "a " "an "])
      name-lower))

;; ---------------------------------------------------------------------------
;; Loading

(defn- fact-args [fact] (mapv #(get % "name") (get fact "arguments")))

(defn- fact-types [fact] (mapv #(get % "type") (get fact "arguments")))

(defn- preconditions [event]
  (mapv (fn [p] [(get p "name") (fact-args p)])
        (get-in event ["condition" "preconditions"] [])))

(def ^:private direction-predicates
  {"north_of" "north" "south_of" "south" "east_of" "east" "west_of" "west"})

(defn- static-maps
  "Room connections, door links and key matches from the world facts."
  [world]
  (reduce
   (fn [m fact]
     (let [name (get fact "name")
           args (fact-args fact)]
       (cond
         ;; north_of(A, B): A is north of B, so going north from B reaches A
         (direction-predicates name)
         (let [[room-to room-from] args]
           (update m :connections
                   (fn [conns]
                     (od-put conns room-from
                             (od-put (od-get conns room-from []) (direction-predicates name) room-to)))))

         (= name "link")
         (let [[room-a door-id room-b] args]
           (update m :door-links od-put [room-a room-b] door-id))

         (= name "match")
         (let [[key-id lockable-id] args]
           (-> m
               (update :key-matches od-put key-id lockable-id)
               (assoc-in [:lockable-key lockable-id] key-id)))

         :else m)))
   {:connections [] :door-links [] :key-matches [] :lockable-key {}}
   world))

(defn- name-index
  "lower-case name (and noun) -> entity ids, in spec order."
  [infos-od]
  (reduce
   (fn [index [eid info]]
     (let [name (get info "name")]
       (if (seq name)
         (let [key (str/lower-case name)
               noun (get info "noun")
               index (update index key (fnil conj []) eid)]
           (if (and (seq noun) (not= (str/lower-case noun) key))
             (update index (str/lower-case noun) (fnil conj []) eid)
             index))
         index)))
   {}
   infos-od))

(defn- parse-spec [spec]
  (let [infos-od (reduce (fn [od [eid info]] (od-put od eid info)) [] (get spec "infos"))
        quest-specs (get spec "quests" [])
        quests (vec (for [quest quest-specs
                          :let [win-events (get quest "win_events" [])
                                pre (if (seq win-events) (preconditions (first win-events)) [])]
                          :when (seq pre)]
                      {:commands (get quest "commands" [])
                       :reward (get quest "reward" 1)
                       :preconditions pre}))
        fail-conditions (vec (for [quest quest-specs
                                   fe (get quest "fail_events" [])
                                   :let [pre (preconditions fe)]
                                   :when (seq pre)]
                               pre))
        objective (get spec "objective" "")
        objective (if (and (not (seq objective)) (seq quests))
                    (get (first quest-specs) "desc" "")
                    objective)]
    (merge
     {:infos (into {} infos-od)
      :name-to-ids (name-index infos-od)
      :objective objective
      :quests quests
      :fail-conditions fail-conditions
      :max-score (reduce + 0 (map :reward quests))
      :world (get spec "world")}
     (static-maps (get spec "world"))
     {:player-room ""
      :inventory #{}
      :obj-locations []
      :obj-in-container []
      :obj-on-supporter []
      :container-state {}
      :door-state {}
      :eaten #{}
      :score 0
      :won false
      :done false
      :moves 0
      :completed-quests 0})))

(defn load-game
  "Load a TextWorld .json game spec; returns the simulator (an atom). Call `reset!` before playing."
  [path]
  (atom (parse-spec (json/read-str (slurp path)))))

;; ---------------------------------------------------------------------------
;; Entity helpers

(defn- info-of [st eid] (get (:infos st) eid {}))

(defn- entity-type [st eid]
  (cond (= eid "P") "P"
        (= eid "I") "I"
        :else (if-let [info (get (:infos st) eid)] (get info "type" "") "")))

(defn- entity-name [st eid]
  (or (not-empty (get (info-of st eid) "name" eid)) eid))

(defn- the-name
  "Inform7 uses the bare name for doors and 'the X' for everything else."
  [st eid]
  (let [name (entity-name st eid)]
    (if (= (entity-type st eid) "d") name (str "the " name))))

(defn- article [name]
  (if (and (seq name) (str/includes? "aeiou" (str/lower-case (subs name 0 1)))) "an" "a"))

(defn- english-list
  "'a X, an Y and a Z'."
  [names]
  (if (empty? names)
    "nothing"
    (let [articles (mapv #(str (article %) " " %) names)]
      (if (= 1 (count articles))
        (first articles)
        (str (str/join ", " (pop articles)) " and " (peek articles))))))

(defn- names-of [st eids] (mapv #(entity-name st %) eids))

(defn- contents [st container-id] (od-keys-where (:obj-in-container st) container-id))

(defn- items-on [st supporter-id] (od-keys-where (:obj-on-supporter st) supporter-id))

(defn- lockable-state
  "Container or door state, or nil for anything else (or an unknown state)."
  [st eid]
  (case (entity-type st eid)
    "c" (get (:container-state st) eid)
    "d" (get (:door-state st) eid)
    nil))

(defn- set-lockable-state [st eid state]
  (if (= (entity-type st eid) "c")
    (assoc-in st [:container-state eid] state)
    (assoc-in st [:door-state eid] state)))

;; ---------------------------------------------------------------------------
;; Facts and quests

(defn- fact-holds? [st pred-name args]
  (case pred-name
    "at" (let [[entity room] args]
           (if (= (entity-type st entity) "P")
             (= (:player-room st) room)
             (= (od-get (:obj-locations st) entity) room)))
    "in" (let [[entity container] args]
           (if (or (= (entity-type st container) "I") (= container "I"))
             (contains? (:inventory st) entity)
             (= (od-get (:obj-in-container st) entity) container)))
    "on" (let [[entity supporter] args]
           (= (od-get (:obj-on-supporter st) entity) supporter))
    ("open" "closed" "locked")
    (let [entity (first args)]
      (case (entity-type st entity)
        "c" (= (get (:container-state st) entity) pred-name)
        "d" (= (get (:door-state st) entity) pred-name)
        false))
    "eaten" (contains? (:eaten st) (first args))
    ("north_of" "south_of" "east_of" "west_of" "free") true
    "link" (if (= 3 (count args))
             (let [[room-a door-id room-b] args]
               (= (od-get (:door-links st) [room-a room-b]) door-id))
             true)
    "match" (if (= 2 (count args))
              (let [[key-id lockable-id] args]
                (= (od-get (:key-matches st) key-id) lockable-id))
              true)
    false))

(defn- preconditions-met? [st pres]
  (every? (fn [[pred-name args]]
            (or (= pred-name "event") (fact-holds? st pred-name args)))
          pres))

(defn- check-quest-progress
  "Complete quests in order, starting from the last one completed."
  [st]
  (let [quests (:quests st)]
    (loop [st st]
      (let [n (:completed-quests st)]
        (if (and (< n (count quests)) (preconditions-met? st (:preconditions (nth quests n))))
          (recur (update st :completed-quests inc))
          st)))))

(defn- fail-condition-met? [st]
  (boolean (some #(preconditions-met? st %) (:fail-conditions st))))

;; ---------------------------------------------------------------------------
;; Inform7 templates

(defn- resolve-entity-ref
  "An entity id, or the first entity with that (lower-case) name."
  [st ref]
  (if (contains? (:infos st) ref)
    ref
    (first (get (:name-to-ids st) (str/lower-case ref)))))

(defn- list-things-in [st container-ref]
  (if-let [cid (resolve-entity-ref st container-ref)]
    (english-list (names-of st (contents st cid)))
    "nothing"))

(defn- list-things-on [st supporter-ref]
  (if-let [sid (resolve-entity-ref st supporter-ref)]
    (english-list (names-of st (items-on st sid)))
    "nothing"))

(defn- state-is?
  "Inform7 state test: a locked thing also counts as closed. Only containers and doors have one."
  [st eid state]
  (let [actual (case (entity-type st eid)
                 "c" (get (:container-state st) eid "closed")
                 "d" (get (:door-state st) eid "closed")
                 nil)]
    (cond (nil? actual) false
          (= state "closed") (contains? #{"closed" "locked"} actual)
          :else (= actual state))))

(defn- eval-single-condition [st cond context]
  (if (and (contains? #{"open" "closed" "locked"} cond) context)
    ;; bare "open"/"closed"/"locked" refers to the entity being described
    (state-is? st context cond)
    (if-let [[_ ref state] (re-find #"^(\w+)\s+is\s+(open|closed|locked)" cond)]
      (if-let [eid (resolve-entity-ref st ref)]
        (state-is? st eid state)
        false)
      (if-let [[_ ref] (re-find #"^there is something in\s+(?:the\s+)?(\w+)" cond)]
        (if-let [eid (resolve-entity-ref st ref)]
          (boolean (seq (contents st eid)))
          false)
        (if-let [[_ ref] (re-find #"^there is something on\s+(?:the\s+)?(\w+)" cond)]
          (if-let [eid (resolve-entity-ref st ref)]
            (boolean (seq (items-on st eid)))
            false)
          (if-let [[_ ref] (re-find #"^there is nothing on\s+(?:the\s+)?(\w+)" cond)]
            (if-let [eid (resolve-entity-ref st ref)]
              (empty? (items-on st eid))
              true)
            (if-let [[_ ref] (re-find #"^(?:the\s+)?(\w+)\s+contains\s+nothing" cond)]
              (if-let [eid (resolve-entity-ref st ref)]
                (empty? (contents st eid))
                true)
              false)))))))

(defn- eval-condition
  "An Inform7 condition, including compound 'A and B'."
  [st condition context]
  (let [cond (str/lower-case (str/trim condition))
        and-parts (re-split #"\s+and\s+" cond)]
    (if (> (count and-parts) 1)
      (every? #(eval-single-condition st (str/trim %) context) and-parts)
      (eval-single-condition st cond context))))

(defn- eval-conditional-block
  "The text of the first true branch of an [if]/[else if]/[otherwise]/[else] block."
  [st condition body context]
  (let [parts (re-split #"(?i)\[(?:else if|otherwise|else)\s*([^\]]*)\]" body)
        n (count parts)
        branches (loop [i 1 branches [[condition (first parts)]]]
                   (cond (>= i n) branches
                         (< (inc i) n) (let [cond (str/trim (nth parts i))]
                                         (recur (+ i 2)
                                                (conj branches [(if (seq cond) cond "__TRUE__")
                                                                (nth parts (inc i))])))
                         :else (recur (inc i) (conj branches ["__TRUE__" (nth parts i)]))))]
    (or (some (fn [[cond text]]
                (when (or (= cond "__TRUE__") (eval-condition st cond context)) text))
              branches)
        "")))

(defn- resolve-inform7-conditionals [st text context]
  (let [text (loop [text text iters 20]
               (if-let [[full condition body] (and (pos? iters)
                                                   (re-find #"(?si)\[if\s+(.+?)\](.*?)\[end if\]" text))]
                 (recur (replace-first-literal text full (eval-conditional-block st condition body context))
                        (dec iters))
                 text))
        list-in (fn [[_ ref]] (list-things-in st ref))
        list-on (fn [[_ ref]] (list-things-on st ref))
        text (-> text
                 (str/replace #"(?i)\[a list of things in the (\w+)\]" list-in)
                 (str/replace #"(?i)\[a list of things in (\w+)\]" list-in)
                 (str/replace #"(?i)\[a list of things on the (\w+)\]" list-on)
                 (str/replace #"(?i)\[a list of things on (\w+)\]" list-on)
                 ;; the Z-machine turns 'word.' into "word." (but not contractions like can't)
                 (str/replace #"(?<=\s)'(\w+[.!?,;]?)'" (fn [[_ word]] (str "\"" word "\""))))]
    ;; Inform7 strips leading whitespace from each line
    (str/trim (str/join "\n" (map str/trim (str/split text #"\n" -1))))))

(defn- process-template
  "Resolve the Inform7 conditionals and lists in a description. `context` is the entity being
  described, for bare conditions like [if open]."
  ([st text] (process-template st text nil))
  ([st text context]
   (if (seq text) (resolve-inform7-conditionals st text context) text)))

;; ---------------------------------------------------------------------------
;; Name matching

(defn- match-entity
  "The candidate best matching a name: exact name (or adj + noun), then noun, then name prefix,
  then noun prefix, then substring. The first candidate wins a tie."
  [st name-lower candidates]
  (when (and (seq name-lower) (seq candidates))
    (let [name-lower (strip-article name-lower)]
      (if (and (contains? (:infos st) name-lower) (some #{name-lower} candidates))
        name-lower
        (:id (reduce
              (fn [best eid]
                (let [info (info-of st eid)
                      ename (str/lower-case (or (get info "name") ""))
                      enoun (str/lower-case (or (get info "noun") ""))
                      eadj (str/lower-case (or (get info "adj") ""))
                      full-name (if (seq eadj) (str/trim (str eadj " " enoun)) enoun)
                      score (cond (or (= ename name-lower) (= full-name name-lower)) 100
                                  (= enoun name-lower) 80
                                  (str/starts-with? ename name-lower) 60
                                  (and (seq enoun) (str/starts-with? enoun name-lower)) 50
                                  (str/includes? ename name-lower) 40
                                  :else 0)]
                  (if (> score (:score best)) {:id eid :score score} best)))
              {:id nil :score 0}
              candidates))))))

(defn- in-room? [st eid] (= (od-get (:obj-locations st) eid) (:player-room st)))

(defn- resolve-in-room
  "Things in the current room: on the floor, fixed in place, or in/on something there."
  [st name]
  (let [name-lower (str/trim (str/lower-case name))]
    (when (seq name-lower)
      (let [room (:player-room st)
            candidates (concat (od-keys-where (:obj-locations st) room)
                               (keep (fn [[eid cid]] (when (in-room? st cid) eid)) (:obj-in-container st))
                               (keep (fn [[eid sid]] (when (in-room? st sid) eid)) (:obj-on-supporter st)))]
        (match-entity st name-lower (vec candidates))))))

(defn- resolve-adjacent-door [st name]
  (let [name-lower (str/trim (str/lower-case name))]
    (when (seq name-lower)
      (let [room (:player-room st)
            doors (distinct (keep (fn [[[ra rb] did]] (when (or (= ra room) (= rb room)) did))
                                  (:door-links st)))]
        (match-entity st name-lower (vec doors))))))

(defn- resolve-in-inventory
  "Carried things. The Python matches against its inventory set in hash order; ties go to the
  first id in sorted order here."
  [st name]
  (let [name-lower (str/trim (str/lower-case name))]
    (when (seq name-lower)
      (match-entity st name-lower (vec (sort (:inventory st)))))))

(defn- resolve-in-room-or-adjacent [st name]
  (or (resolve-in-room st name) (resolve-adjacent-door st name)))

(defn- resolve-target [st name]
  (or (resolve-in-room st name) (resolve-in-inventory st name) (resolve-adjacent-door st name)))

(defn- find-among [st obj-name candidates]
  (match-entity st (strip-article (str/trim (str/lower-case obj-name))) (vec candidates)))

;; ---------------------------------------------------------------------------
;; Look

(defn- floor-objects
  "Portable things lying in a room (not in or on anything)."
  [st room-id]
  (for [[eid loc] (:obj-locations st)
        :when (and (= loc room-id)
                   (contains? #{"o" "f" "k"} (entity-type st eid))
                   (not (od-has? (:obj-in-container st) eid))
                   (not (od-has? (:obj-on-supporter st) eid)))]
    eid))

(defn- look [st]
  (let [room-info (get (:infos st) (:player-room st) {})
        room-name (get room-info "name" "unknown room")
        desc (process-template st (get room-info "desc" ""))
        names (names-of st (floor-objects st (:player-room st)))
        lines [(str "-= " (title-case room-name) " =-") desc]
        lines (cond (empty? names) lines
                    (= 1 (count names)) (conj lines (str "\nThere is " (article (first names)) " "
                                                         (first names) " on the floor."))
                    :else (conj lines (str "\nThere is " (english-list names) " on the floor.")))]
    (str/join "\n" lines)))

;; ---------------------------------------------------------------------------
;; Commands. Each takes the state and returns [state message].

(def ^:private no-such-thing "You can't see any such thing.")

(defn- go [st direction]
  (let [room (:player-room st)
        exits (od-get (:connections st) room [])]
    (if-not (od-has? exits direction)
      [st "You can't go that way."]
      (let [target (od-get exits direction)
            door-id (or (od-get (:door-links st) [room target])
                        (od-get (:door-links st) [target room]))]
        (if (and door-id (contains? #{"locked" "closed"} (get (:door-state st) door-id "open")))
          [st (str "You have to open the " (entity-name st door-id) " first.")]
          (let [st (assoc st :player-room target)]
            [st (look st)]))))))

(defn- inventory [st]
  [st (if (empty? (:inventory st))
        "You are carrying nothing."
        (str "You are carrying: " (english-list (names-of st (sort (:inventory st)))) "."))])

(defn- examine [st target]
  (if-let [eid (resolve-target st target)]
    (let [desc (get (info-of st eid) "desc" "")
          desc (if (seq desc) (process-template st desc eid) desc)
          name (entity-name st eid)]
      [st (case (entity-type st eid)
            "c" (let [state (get (:container-state st) eid "closed")
                      desc (if (seq desc) desc (str "The " name " is " state "."))]
                  (if (= state "open")
                    (let [inside (contents st eid)]
                      (if (seq inside)
                        (str desc "\nThe " name " contains " (english-list (names-of st inside)) ".")
                        (str desc "\nThe " name " is empty.")))
                    desc))
            "s" (let [desc (if (seq desc) desc (str "The " name " is here."))
                      on (items-on st eid)]
                  (if (seq on)
                    (str desc "\nOn the " name " you can see " (english-list (names-of st on)) ".")
                    (str desc "\nThe " name " is empty.")))
            "d" (if (seq desc) desc (str "The " name " is " (get (:door-state st) eid "closed") "."))
            (if (seq desc) desc (str "You see nothing special about the " name ".")))])
    [st no-such-thing]))

(defn- pick-up [st eid]
  (update st :inventory conj eid))

(defn- take-thing [st target]
  (let [eid (resolve-in-room st target)]
    (cond
      (nil? eid) [st no-such-thing]
      (not (contains? #{"o" "f" "k"} (entity-type st eid)))
      [st (str "You can't take the " (entity-name st eid) ".")]
      :else
      (let [name (entity-name st eid)
            sup-id (od-get (:obj-on-supporter st) eid)
            cont-id (od-get (:obj-in-container st) eid)]
        (cond
          ;; on the floor
          (and (in-room? st eid) (nil? cont-id) (nil? sup-id))
          [(-> st (update :obj-locations od-del eid) (pick-up eid))
           (str "You pick up the " name " from the ground.")]

          ;; on a supporter here
          (and sup-id (in-room? st sup-id))
          [(-> st (update :obj-on-supporter od-del eid) (pick-up eid))
           (str "You take the " name " from the " (entity-name st sup-id) ".")]

          ;; in a container here
          (and cont-id (in-room? st cont-id))
          (let [state (get (:container-state st) cont-id "closed")]
            (if (= state "open")
              [(-> st (update :obj-in-container od-del eid) (pick-up eid))
               (str "You take the " name " from the " (entity-name st cont-id) ".")]
              [st (str "The " (entity-name st cont-id) " is " state ".")]))

          :else [st (str "You can't take the " name ".")])))))

(defn- take-from [st obj-name source-name]
  (if-let [source-id (resolve-in-room st source-name)]
    (let [source (entity-name st source-id)]
      (case (entity-type st source-id)
        "s" (if-let [obj-id (find-among st obj-name (items-on st source-id))]
              [(-> st (update :obj-on-supporter od-del obj-id) (pick-up obj-id))
               (str "You take the " (entity-name st obj-id) " from the " source ".")]
              [st no-such-thing])
        "c" (let [state (get (:container-state st) source-id "closed")]
              (if (not= state "open")
                [st (str "The " source " is " state ".")]
                (if-let [obj-id (find-among st obj-name (contents st source-id))]
                  [(-> st (update :obj-in-container od-del obj-id) (pick-up obj-id))
                   (str "You take the " (entity-name st obj-id) " from the " source ".")]
                  [st no-such-thing])))
        [st (str "You can't take things from the " source ".")]))
    [st no-such-thing]))

(defn- parse-take [st args]
  (if-let [[_ obj source] (re-find #"(?i)^(.+?)\s+from\s+(.+)" args)]
    (take-from st (str/trim obj) (str/trim source))
    (take-thing st args)))

(defn- drop-thing [st target]
  (if-let [eid (resolve-in-inventory st target)]
    [(-> st
         (update :inventory disj eid)
         (update :obj-locations od-put eid (:player-room st)))
     (str "You drop the " (entity-name st eid) " on the ground.")]
    [st no-such-thing]))

(defn- open-thing [st target]
  (if-let [eid (resolve-in-room-or-adjacent st target)]
    (let [name (entity-name st eid)]
      (case (entity-type st eid)
        "c" (if-not (in-room? st eid)
              [st no-such-thing]
              (case (get (:container-state st) eid "closed")
                "open" [st (str "The " name " is already open.")]
                "locked" [st (str "The " name " is locked.")]
                (let [inside (contents st eid)]
                  [(assoc-in st [:container-state eid] "open")
                   ;; Inform7 says "revealing", not "contains"
                   (if (seq inside)
                     (str "You open the " name ", revealing " (english-list (names-of st inside)) ".")
                     (str "You open the " name "."))])))
        "d" (case (get (:door-state st) eid "closed")
              "open" [st "That's already open."]
              "locked" [st "It is locked."]
              [(assoc-in st [:door-state eid] "open") (str "You open " (the-name st eid) ".")])
        [st (str "You can't open " (the-name st eid) ".")]))
    [st no-such-thing]))

(defn- close-thing [st target]
  (if-let [eid (resolve-in-room-or-adjacent st target)]
    (let [name (entity-name st eid)]
      (case (entity-type st eid)
        "c" (if-not (in-room? st eid)
              [st no-such-thing]
              (case (get (:container-state st) eid "closed")
                "closed" [st (str "The " name " is already closed.")]
                "locked" [st (str "The " name " is locked.")]
                [(assoc-in st [:container-state eid] "closed") (str "You close the " name ".")]))
        "d" (case (get (:door-state st) eid "closed")
              "closed" [st "That's already closed."]
              "locked" [st "It is locked."]
              [(assoc-in st [:door-state eid] "closed") (str "You close " (the-name st eid) ".")])
        [st (str "You can't close " (the-name st eid) ".")]))
    [st no-such-thing]))

(defn- wrong-key [st key-id target-id]
  (str "The " (entity-name st key-id) " doesn't fit the " (entity-name st target-id) "."))

(defn- unlock [st target-name key-name]
  (let [target-id (resolve-in-room-or-adjacent st target-name)
        key-id (when target-id (resolve-in-inventory st key-name))]
    (cond
      (or (nil? target-id) (nil? key-id)) [st no-such-thing]
      (not (contains? #{"c" "d"} (entity-type st target-id)))
      [st (str "You can't unlock " (the-name st target-id) ".")]
      (not= (lockable-state st target-id) "locked")
      [st (str "The " (entity-name st target-id) " is not locked.")]
      (not= (get (:lockable-key st) target-id) key-id) [st (wrong-key st key-id target-id)]
      :else [(set-lockable-state st target-id "closed") (str "You unlock " (the-name st target-id) ".")])))

(defn- lock [st target-name key-name]
  (let [target-id (resolve-in-room-or-adjacent st target-name)
        key-id (when target-id (resolve-in-inventory st key-name))]
    (if (or (nil? target-id) (nil? key-id))
      [st no-such-thing]
      (if-not (contains? #{"c" "d"} (entity-type st target-id))
        [st (str "You can't lock " (the-name st target-id) ".")]
        (let [state (lockable-state st target-id)
              name (entity-name st target-id)]
          (cond
            (= state "locked") [st (str "The " name " is already locked.")]
            (= state "open") [st (str "First you would have to close " (the-name st target-id) ".")]
            (not= state "closed") [st (str "The " name " is not closed.")]
            (not= (get (:lockable-key st) target-id) key-id) [st (wrong-key st key-id target-id)]
            :else [(set-lockable-state st target-id "locked") (str "You lock " (the-name st target-id) ".")]))))))

(defn- with-two-names
  "Parse '<a> <word> <b>' and call f with the two names, or answer with the usage hint."
  [st args re usage f]
  (if-let [[_ a b] (re-find re args)]
    (f st (str/trim a) (str/trim b))
    [st usage]))

(defn- put-on [st obj-name sup-name]
  (let [obj-id (resolve-in-inventory st obj-name)
        sup-id (when obj-id (resolve-in-room st sup-name))]
    (cond
      (or (nil? obj-id) (nil? sup-id)) [st no-such-thing]
      (not= (entity-type st sup-id) "s") [st (str "You can't put things on the " (entity-name st sup-id) ".")]
      :else [(-> st
                 (update :inventory disj obj-id)
                 (update :obj-on-supporter od-put obj-id sup-id))
             (str "You put the " (entity-name st obj-id) " on the " (entity-name st sup-id) ".")])))

(defn- insert-into [st obj-name cont-name]
  (let [obj-id (resolve-in-inventory st obj-name)
        cont-id (when obj-id (resolve-in-room st cont-name))]
    (cond
      (or (nil? obj-id) (nil? cont-id)) [st no-such-thing]
      (not= (entity-type st cont-id) "c")
      [st (str "You can't insert things into the " (entity-name st cont-id) ".")]
      :else
      (let [state (get (:container-state st) cont-id "closed")]
        (if (not= state "open")
          [st (str "The " (entity-name st cont-id) " is " state ".")]
          [(-> st
               (update :inventory disj obj-id)
               (update :obj-in-container od-put obj-id cont-id))
           ;; Inform7 says "put into", not "insert into"
           (str "You put the " (entity-name st obj-id) " into the " (entity-name st cont-id) ".")])))))

(defn- eat [st target]
  (if-let [eid (resolve-in-inventory st target)]
    (if (not= (entity-type st eid) "f")
      [st (str "You can't eat the " (entity-name st eid) ".")]
      [(-> st (update :inventory disj eid) (update :eaten conj eid))
       (str "You eat the " (entity-name st eid) ". Not bad.")])
    [st no-such-thing]))

(defn- examine-target
  "Python's command.split(None, 1)[1]: everything after the first run of whitespace."
  [command]
  (if (str/includes? command " ")
    (or (second (re-find #"(?s)^\s*\S+\s+(.*)$" command)) "")
    ""))

(defn- execute [st command]
  (let [cmd (str/trim (str/lower-case command))
        rest-of (fn [n] (str/trim (subs command n)))
        starts? (fn [prefix] (str/starts-with? cmd prefix))]
    (cond
      (contains? #{"go north" "north" "n"} cmd) (go st "north")
      (contains? #{"go south" "south" "s"} cmd) (go st "south")
      (contains? #{"go east" "east" "e"} cmd) (go st "east")
      (contains? #{"go west" "west" "w"} cmd) (go st "west")
      (contains? #{"look" "l"} cmd) [st (look st)]
      (contains? #{"inventory" "i"} cmd) (inventory st)
      (or (starts? "examine ") (starts? "x ")) (examine st (examine-target command))
      (starts? "take ") (parse-take st (rest-of 5))
      (starts? "drop ") (drop-thing st (rest-of 5))
      (starts? "open ") (open-thing st (rest-of 5))
      (starts? "close ") (close-thing st (rest-of 6))
      (starts? "unlock ")
      (with-two-names st (rest-of 7) #"(?i)^(.+?)\s+with\s+(.+)"
        "You need to specify what to unlock it with. Try: unlock <thing> with <key>" unlock)
      (starts? "lock ")
      (with-two-names st (rest-of 5) #"(?i)^(.+?)\s+with\s+(.+)"
        "You need to specify what to lock it with. Try: lock <thing> with <key>" lock)
      (starts? "put ")
      (with-two-names st (rest-of 4) #"(?i)^(.+?)\s+on\s+(.+)"
        "You need to specify where to put it. Try: put <item> on <supporter>" put-on)
      (starts? "insert ")
      (with-two-names st (rest-of 7) #"(?i)^(.+?)\s+into\s+(.+)"
        "You need to specify where to insert it. Try: insert <item> into <container>" insert-into)
      (starts? "eat ") (eat st (rest-of 4))
      :else [st "I don't understand that command."])))

;; ---------------------------------------------------------------------------
;; Game loop

(defn- make-info [st]
  {"score" (:score st)
   "max_score" (:max-score st)
   "won" (:won st)
   "moves" (:moves st)})

(defn- initial-state
  "The mutable part of the state, loaded from the world facts."
  [st]
  (reduce
   (fn [st fact]
     (let [[a b] (fact-args fact)
           [ta tb] (fact-types fact)]
       (case (get fact "name")
         "at" (cond (= ta "P") (assoc st :player-room b)
                    ;; containers and supporters are fixed in place but are "at" a room too
                    (contains? #{"o" "f" "k" "c" "s"} ta) (update st :obj-locations od-put a b)
                    :else st)
         "in" (case tb
                "I" (update st :inventory conj a)
                "c" (update st :obj-in-container od-put a b)
                st)
         "on" (update st :obj-on-supporter od-put a b)
         ("open" "closed" "locked")
         (case ta
           "c" (assoc-in st [:container-state a] (get fact "name"))
           "d" (assoc-in st [:door-state a] (get fact "name"))
           st)
         st)))
   (assoc st
          :player-room ""
          :inventory #{}
          :obj-locations []
          :obj-in-container []
          :obj-on-supporter []
          :container-state {}
          :door-state {}
          :eaten #{}
          :score 0
          :won false
          :done false
          :moves 1 ; Inform7 counts the opening look as turn 1
          :completed-quests 0)
   (:world st)))

(defn reset!
  "Reset the game to its initial state. Returns [observation info]."
  [sim]
  (let [st (swap! sim initial-state)]
    [(str/join "\n" [(:objective st) "" (look st)]) (make-info st)]))

(defn- score-quests [st]
  (let [prev-score (:score st)
        st (check-quest-progress st)
        n-quests (count (:quests st))
        score (reduce + 0 (map :reward (take (:completed-quests st) (:quests st))))
        st (assoc st :score score)
        gained (when (> score prev-score)
                 (let [delta (- score prev-score)]
                   (str "\n\n\nYour score has just gone up by "
                        (if (== delta 1) "one point" (str delta " points")) ".")))]
    (cond
      (and (>= (:completed-quests st) n-quests) (pos? n-quests))
      (let [st (assoc st :won true :done true)]
        [st (str gained "\n\n\n"
                 "                                                       "
                 "*** The End ***\n\n"
                 "You scored " (:score st) " out of a possible " (:max-score st)
                 ", in " (:moves st) " turns.")])
      (fail-condition-met? st) [(assoc st :done true) (str gained "\n\n*** You lost! ***")]
      :else [st gained])))

(defn- step-state [st command]
  (if (:done st)
    [st "The game has already ended." true]
    (let [st (update st :moves inc)
          command (let [c (str/trim command)] (if (seq c) c "look"))
          [st obs] (execute st command)
          [st extra] (if (or (:won st) (:done st)) [st nil] (score-quests st))]
      [st (str obs extra) false])))

(defn step!
  "Play one command. Returns [observation reward done info]; the reward is the current score
  (the total reward of the quests completed so far), as in TextWorld."
  [sim command]
  (let [result (volatile! nil)]
    (swap! sim (fn [st]
                 (let [[st obs ended?] (step-state st command)]
                   (vreset! result [obs (if ended? 0.0 (double (:score st))) (:done st) (make-info st)])
                   st)))
    @result))

;; ---------------------------------------------------------------------------
;; Accessors

(defn score [sim] (:score @sim))

(defn won? [sim] (:won @sim))

(defn done? [sim] (:done @sim))

(defn moves [sim] (:moves @sim))

(defn feedback
  "The Python property reads `_last_obs`, which the simulator never sets, so this is always \"\"."
  [_sim]
  "")

(defn privileged-state
  "The full game state in readable form (for oracle annotations), with the string keys of the
  Python `get_privileged_state()`."
  [sim]
  (let [st @sim
        room (:player-room st)
        infos (:infos st)
        name-or (fn [eid] (get (get infos eid {}) "name" eid))
        exits (into {}
                    (for [[direction dest] (od-get (:connections st) room [])
                          :let [door-id (or (od-get (:door-links st) [room dest])
                                            (od-get (:door-links st) [dest room]))]]
                      [direction (if door-id
                                   {"destination" (name-or dest)
                                    "door" (entity-name st door-id)
                                    "door_state" (get (:door-state st) door-id "open")}
                                   {"destination" (name-or dest)})]))
        room-objects (vec (for [[eid loc] (:obj-locations st)
                                :when (= loc room)
                                :let [etype (entity-type st eid)
                                      entry {"name" (entity-name st eid) "type" etype}]]
                            (case etype
                              ("o" "f" "k")
                              (assoc entry "location"
                                     (if-let [cid (od-get (:obj-in-container st) eid)]
                                       (str "in " (entity-name st cid))
                                       (if-let [sid (od-get (:obj-on-supporter st) eid)]
                                         (str "on " (entity-name st sid))
                                         "floor")))
                              "c" (assoc entry
                                         "state" (get (:container-state st) eid "closed")
                                         "contents" (names-of st (contents st eid)))
                              "s" (assoc entry "items_on" (names-of st (items-on st eid)))
                              entry)))
        room-map (reduce (fn [m [room-id conns]]
                           (assoc m (name-or room-id)
                                  (into {} (for [[d dest] conns] [d (name-or dest)]))))
                         {}
                         (:connections st))]
    {"player_room" (name-or room)
     "exits" exits
     "inventory" (vec (sort (names-of st (:inventory st))))
     "room_objects" room-objects
     "room_map" room-map
     "key_locks" (mapv (fn [[key-id lockable-id]]
                         {"key" (entity-name st key-id) "unlocks" (entity-name st lockable-id)})
                       (:key-matches st))
     "walkthrough" (vec (mapcat :commands (:quests st)))
     "score" (:score st)
     "max_score" (:max-score st)
     "completed_quests" (:completed-quests st)
     "total_quests" (count (:quests st))
     "objective" (:objective st)}))
