(ns distilla.rl.grpo
  "Reinforcement learning of the memory module on TextWorld with GRPO, as
  memorilla's rl/ does on SkyRL-Memorilla, in one process.

  Rollouts (SkyRL's multi-turn, token-in-token-out agent loop):
  - the frozen agent reads the conversation (system prompt, opening
    observation, then every reply and observation) and ends each turn with
    [ACTION: <command>]; replies are appended as generated ids (stop token
    included, loss mask 1), observations as chat-template ids (mask 0);
  - once the environment has memory documents, the frozen encoder embeds
    them, the memory module compresses them (conditioned on the mean
    document embedding) and the K memory vectors sit before the prompt; they
    are recomputed at every turn;
  - a turn's reward is the game score TextWorld reports; a rollout's return
    is the sum.

  Training: GRPO with group-normalised advantages ((R - mean) / (std + 1e-6)
  over the samples of a prompt), one update per batch (so the PPO ratio is 1
  and clipping is inactive), token-mean loss over micro-batches, no KL or
  entropy term, AdamW at a constant rate, gradient clipping at 1. As in the
  SkyRL training pass, each trajectory is scored as one sequence with the
  memory computed from its final documents. Only the memory module learns."
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [distilla.encoder :as encoder]
            [distilla.memory :as memory]
            [distilla.optim :as optim]
            [distilla.rl.env :as env]
            [distilla.tensor :as t]
            [distilla.tokenizer :as tok]
            [distilla.util :as util]))

(def projection-init-std 0.01)

;; --- the policy -------------------------------------------------------------------------------

(defn init-memory-module
  "A fresh module as the SkyRL handler builds one: torch's initialisation,
  then the output projection redrawn from N(0, 0.01) with zero bias, so the
  first memory vectors are small."
  [cfg seed]
  (let [m (memory/init-module cfg seed)
        state (memory/rng (+ seed 7919))
        n (get-in m [:params "memory_projection.weight" :n])
        normal (fn [] (* (Math/sqrt (* -2.0 (Math/log (max 1e-300 (memory/uniform! state)))))
                         (Math/cos (* 2.0 Math/PI (memory/uniform! state)))))]
    (memory/set-param! m "memory_projection.weight" (float-array (repeatedly n #(* projection-init-std (normal)))))
    (memory/set-param! m "memory_projection.bias" (float-array (get-in m [:params "memory_projection.bias" :n])))
    m))

(defn memory-vectors
  "The memory vectors for a list of documents: embedded, compressed with the
  mean document embedding as the query. nil without documents."
  [{:keys [memory-module encoder]} documents]
  (when (and memory-module (seq documents))
    (let [docs (encoder/embed-texts encoder documents)
          query (let [q (float-array (:cols docs))]
                  (dotimes [i (:rows docs)]
                    (dotimes [j (:cols docs)]
                      (aset q j (float (+ (aget q j) (/ (aget ^floats (:data docs) (+ (* i (:cols docs)) j)) (:rows docs)))))))
                  q)]
      {:docs docs :query query})))

;; --- tokens --------------------------------------------------------------------------------------

(def base-conversation [{:role "system" :content "You are a helpful assistant."} {:role "user" :content "I am a user."}])

(defn- base-ids
  "The base conversation's ids up to its last EOS (SkyRL's fixed-base trick)."
  [tokenizer eos]
  (let [ids (tok/encode tokenizer (tok/apply-chat-template tokenizer base-conversation false))
        last-eos (.lastIndexOf ^java.util.List ids eos)]
    (if (neg? last-eos) ids (subvec ids 0 (inc last-eos)))))

(defn observation-ids
  "The ids of observation messages in chat-template form, ending with the next
  generation prompt."
  [tokenizer base observations]
  (let [ids (tok/encode tokenizer (tok/apply-chat-template tokenizer (into base-conversation observations) true))]
    (subvec ids (count base))))

;; --- rollouts ------------------------------------------------------------------------------------

(declare finish)

(defn rollout
  "Play one episode. policy: {:generate fn [segments opts] -> {:tokens :text
  :stop-token :stop-reason}, :tokenizer, :eos, :memory-module, :encoder,
  :num-memories}. row: a dataset row (prompt, env_class, game_file, ...).
  opts: :max-turns :max-input-length :max-new-tokens :temperature :seed
  :enable-memory and engine options.
  Returns the trajectory: :prompt-ids :response-ids :loss-mask :rewards
  (per turn) :token-rewards :return :stop-reason :documents :metrics."
  [{:keys [generate tokenizer eos num-memories memory-module] :as policy} row
   {:keys [max-turns max-input-length max-new-tokens temperature seed enable-memory]
    :or {max-turns 50 max-input-length 16000 max-new-tokens 128 temperature 1.0 seed 0 enable-memory true}
    :as opts}]
  (let [e (env/make-env (merge opts {:env-class (get row "env_class") :game-file (get row "game_file")
                                     :max-turns max-turns :enable-memory (and enable-memory (some? memory-module))}))
        prompt (mapv (fn [m] {:role (get m "role") :content (get m "content")}) (get row "prompt"))
        base (base-ids tokenizer eos)]
    (try
      (let [messages (env/init! e prompt)
            input-ids (tok/encode tokenizer (tok/apply-chat-template tokenizer messages true))
            prompt-len (count input-ids)]
        (loop [turn 0 ids input-ids mask [] rewards [] mem-k 0 stop-reason "stop"]
          (let [docs (env/documents e)
                mem-k (if (and (zero? mem-k) (seq docs)) num-memories mem-k)]
            (if (> (+ mem-k (count ids)) max-input-length)
              (finish e base prompt-len ids mask rewards "length" mem-k)
              (let [mv (when (seq docs) (memory-vectors policy docs))
                    memory-out (when mv (memory/forward memory-module (:docs mv) (:query mv)))
                    segments (cond-> [] memory-out (conj [:embeddings memory-out]) true (conj [:tokens ids]))
                    {:keys [tokens text stop-token] gen-stop :stop-reason}
                    (generate segments {:max-new-tokens max-new-tokens :temperature temperature
                                        :seed (+ seed (* 7919 turn))})
                    out-ids (cond-> (vec tokens) stop-token (conj stop-token))
                    {:keys [observations reward done]} (env/step! e text)
                    ids (into ids out-ids)
                    mask (into mask (repeat (count out-ids) 1))
                    end-idx (dec (count ids))
                    rewards (conj rewards [reward end-idx])
                    [ids mask] (if (seq observations)
                                 (let [obs (observation-ids tokenizer base observations)]
                                   [(into ids obs) (into mask (repeat (count obs) 0))])
                                 [ids mask])]
                (if done
                  (finish e base prompt-len ids mask rewards gen-stop mem-k)
                  (recur (inc turn) ids mask rewards mem-k gen-stop)))))))
      (finally (env/close! e)))))

(defn- finish [e _base prompt-len ids mask rewards stop-reason _mem-k]
  (let [response (subvec ids prompt-len)
        token-rewards (let [a (double-array (count response))]
                        (doseq [[r idx] rewards
                                :let [i (- idx prompt-len)]
                                :when (< -1 i (count response))]
                          (aset a i (+ (aget a i) r)))
                        (vec a))
        docs (env/documents e)]
    {:prompt-ids (subvec ids 0 prompt-len)
     :response-ids response
     :loss-mask mask
     :rewards (mapv first rewards)
     :token-rewards token-rewards
     :return (reduce + 0.0 (map first rewards))
     :stop-reason stop-reason
     :documents docs
     :metrics (env/metrics e)}))

;; --- GRPO -------------------------------------------------------------------------------------------

(defn advantages
  "Group-normalised advantages: (R - mean) / (std + eps) within each group
  (std unbiased; a group of one gets mean 0, std 1)."
  ([returns group-size] (advantages returns group-size 1e-6))
  ([returns group-size eps]
   (vec (mapcat (fn [group]
                  (let [n (count group)
                        mean (if (> n 1) (/ (reduce + group) n) 0.0)
                        std (if (> n 1)
                              (Math/sqrt (/ (reduce + (map #(Math/pow (- % mean) 2) group)) (dec n)))
                              1.0)]
                    (map #(/ (- % mean) (+ std eps)) group)))
                (partition-all group-size returns)))))

(defn training-sequence
  "A trajectory as the decoder sees it in training: K memory positions before
  the prompt when it ends with documents, scored on its response tokens."
  [{:keys [prompt-ids response-ids loss-mask documents]} num-memories]
  (let [k (if (seq documents) num-memories 0)
        tokens (into (vec (repeat k 0)) (into prompt-ids response-ids))
        n (count tokens)
        mask (into (vec (repeat (+ k (count prompt-ids)) 0)) loss-mask)]
    {:tokens tokens
     :mem-index (into (vec (range k)) (repeat (- n k) -1))
     :targets (vec (for [t (range n)]
                     (if (and (< (inc t) n) (= 1 (nth mask (inc t)))) (nth tokens (inc t)) -1)))
     :n-response (count (filter #{1} loss-mask))}))

(defn policy-gradient!
  "Accumulate the GRPO gradient of a batch of trajectories into the memory
  module: token-mean loss sum(A * CE) / tokens within each micro-batch,
  averaged over micro-batches. decoder: fn [sequence grad?] -> {:loss
  :n-targets :grad}. Returns the mean policy loss."
  [{:keys [memory-module encoder num-memories rng] :as policy} decoder trajectories advs micro-batch]
  (let [micros (partition-all micro-batch (map vector trajectories advs))
        n-micro (count micros)]
    (util/mean
     (for [micro micros]
       (let [total (reduce + 0 (map (comp :n-response #(training-sequence (first %) num-memories)) micro))]
         (reduce
          +
          0.0
          (for [[traj a] micro
                :let [{:keys [n-response] :as sequence} (training-sequence traj num-memories)]
                :when (and (pos? n-response) (seq (:documents traj)) (not (zero? a)))]
            (let [mv (memory-vectors policy (:documents traj))
                  {:keys [output tape]} (memory/forward-tape memory-module (:docs mv) (:query mv) {:train? true :rng rng})
                  {:keys [loss grad n-targets]} (decoder (assoc sequence :memory output) true)
                  w (* a (/ (double n-targets) (max total 1)) (/ 1.0 n-micro))
                  d (t/copy-floats (:data grad))]
              (dotimes [i (alength d)] (aset d i (float (* w (aget d i)))))
              (memory/backward! memory-module tape (t/matrix (:rows grad) (:cols grad) d))
              (* w loss)))))))))

(defn train-step!
  "One GRPO step on a batch of rows: n-samples rollouts each, advantages per
  row, one optimizer update. Returns step statistics and the trajectories."
  [{:keys [memory-module optimizer] :as policy} decoder rows
   {:keys [n-samples micro-batch lr max-grad-norm sampling] :or {micro-batch 2 max-grad-norm 1.0}}]
  (let [trajectories (vec (for [[i row] (map-indexed vector rows)
                                s (range n-samples)]
                            (rollout policy row (assoc sampling :seed (+ (* 1000003 @(:step optimizer)) (* 1009 i) s)))))
        returns (mapv :return trajectories)
        advs (advantages returns n-samples)]
    (memory/zero-grads! memory-module)
    (let [loss (policy-gradient! policy decoder trajectories advs micro-batch)
          norm (optim/clip-grad-norm! memory-module max-grad-norm)
          step (optim/step! optimizer lr)]
      {:step step :policy-loss loss :grad-norm norm
       :mean-return (util/mean returns)
       :solve-rate (util/mean (map #(if (pos? (or (peek (:rewards %)) 0.0)) 1.0 0.0) trajectories))
       :mean-turns (util/mean (map #(get-in % [:metrics "steps"]) trajectories))
       :trajectories trajectories})))

;; --- evaluation dumps (SkyRL's layout, read by distilla.rl.aggregate) ----------------------------------

(defn dump-record [tokenizer row traj]
  {"input_prompt" (tok/decode tokenizer (:prompt-ids traj))
   "output_response" (tok/decode tokenizer (:response-ids traj))
   "score" (:token-rewards traj)
   "stop_reason" (:stop-reason traj)
   "data_source" (get row "data_source")
   "env_class" (get row "env_class")
   "env_extras" {"game_file" (get row "game_file") "extra_info" (get row "extra_info")}
   "env_metrics" (:metrics traj)})

(defn evaluate!
  "Play every row n-samples times and write one JSONL per tier under
  dump-dir. Returns the records."
  [policy rows {:keys [n-samples sampling dump-dir]}]
  (let [records (vec (for [[i row] (map-indexed vector rows)
                           s (range n-samples)]
                       (dump-record (:tokenizer policy) row
                                    (rollout policy row (assoc sampling :seed (+ (* 1009 i) s))))))]
    (when dump-dir
      (doseq [[tier rs] (group-by #(get % "data_source") records)]
        (util/write-jsonl (fs/path dump-dir (str tier ".jsonl")) rs)))
    records))
