(ns distilla.train
  "Training the memory module against a frozen decoder: train.py's recipe
  runner, and a runtime API for tuning a module on the fly.

  The loss is the decoder's next-token cross-entropy over the answer tokens
  (the prompt carries the memory tokens in its system turn). The decoder is
  frozen: its backward pass (native/distilla_train.c) only delivers the
  gradient with respect to the memory tokens, which the memory module's own
  backward pass (distilla.memory/backward!) turns into parameter gradients.
  AdamW, gradient clipping at norm 1 and the warmup + cosine schedule follow
  train.py.

  Runtime use:

    (def tr (trainer {:memory module :decoder (llama-decoder grad-model)
                      :tokenizer tok :learning-rate 1e-4}))
    (learn! tr encoder [{:question q :answer a :documents docs} ...] {:steps 20})
    (snapshot tr) / (restore! tr snap)   ; try an update, keep it or roll it back"
  (:require [babashka.fs :as fs]
            [clojure.string :as str]
            [distilla.benchmarks :as b]
            [distilla.cli :as cli]
            [distilla.data :as data]
            [distilla.dataset :as ds]
            [distilla.embeddings :as emb]
            [distilla.encoder :as encoder]
            [distilla.memory :as memory]
            [distilla.optim :as optim]
            [distilla.paths :as paths]
            [distilla.store :as store]
            [distilla.tensor :as t]
            [distilla.tokenizer :as tok]
            [distilla.util :as util]))

(def gradient-clip-val 1.0)
(def epoch-dir-template "epoch-%02d")
(def last-dir "last")
(def log-file "logs/metrics.csv")

;; --- sequences ---------------------------------------------------------------------------------

(defn sequence-inputs
  "The decoder inputs of one collated row (padding removed): token ids with
  the memory positions numbered in order, and the token each position must
  predict (labels shifted by one, -1 where unsupervised)."
  [ids labels memory-id]
  (let [n (count ids)
        mem-index (second (reduce (fn [[k acc] id] (if (= id memory-id) [(inc k) (conj acc k)] [k (conj acc -1)]))
                                  [0 []] ids))]
    {:tokens (mapv #(if (= % memory-id) 0 %) ids)
     :mem-index mem-index
     :targets (vec (for [i (range n)]
                     (let [l (when (< (inc i) n) (nth labels (inc i)))]
                       (if (or (nil? l) (= l data/ignore-index)) -1 l))))}))

(defn llama-decoder
  "The decoder seam of a trainer over a distilla.llama grad-model: a fn of
  [sequence grad?] returning {:loss :n-targets :grad}."
  [grad-model]
  (let [f (requiring-resolve 'distilla.llama/loss-and-grad)]
    (fn [sequence grad?] (f grad-model sequence :grad? grad?))))

;; --- the trainer --------------------------------------------------------------------------------

(defn trainer
  "A training state around a memory module.
  opts:
    :memory        the module (trained in place)
    :decoder       fn [sequence grad?] -> {:loss :n-targets :grad} (llama-decoder)
    :tokenizer     the decoder's tokenizer
    :system-prompt :suffix :max-length   prompt settings (data/collate)
    :learning-rate (1e-4) :weight-decay (0.01) :max-grad-norm (1.0)
    :schedule      fn optimizer-step -> lr (default: constant learning-rate)
    :seed          dropout seed"
  [{:keys [memory decoder tokenizer system-prompt suffix max-length learning-rate weight-decay max-grad-norm
           schedule seed]
    :or {system-prompt data/default-system-prompt suffix "" max-length 2048 learning-rate 1e-4
         weight-decay 0.01 max-grad-norm gradient-clip-val seed util/default-seed}}]
  (memory/ensure-grads! memory)
  {:memory memory
   :decoder decoder
   :tokenizer tokenizer
   :collate {:tokenizer tokenizer :num-memories (memory/num-memories memory) :max-length max-length
             :system-prompt system-prompt :suffix suffix :train true}
   :optimizer (optim/adamw memory {:weight-decay weight-decay})
   :schedule (atom (or schedule (optim/constant-schedule learning-rate)))
   :max-grad-norm max-grad-norm
   :rng (memory/rng seed)})

(defn collate
  "Collate items (rows of a memory dataset, or examples carrying their own
  embeddings) in training mode."
  [tr items stores]
  (data/collate (assoc (:collate tr) :stores stores) items))

(defn batch-loss!
  "The loss of a collated batch, as the Python module computes it: the mean
  cross-entropy over every supervised token of the batch. With :train? the
  gradients of (scale x loss) are accumulated into the module."
  [{:keys [memory decoder tokenizer rng]} batch {:keys [train? scale] :or {scale 1.0}}]
  (let [{:keys [input-ids attention-mask labels doc-embeds doc-padding-mask question-embeds]} batch
        mem-id (tok/memory-token-id tokenizer)
        rows (mapv (fn [ids mask lab docs pad q]
                     (let [keep (fn [xs] (vec (keep-indexed (fn [i x] (when (pos? (nth mask i)) x)) xs)))
                           seq-in (sequence-inputs (keep ids) (keep lab) mem-id)
                           {:keys [output tape]} (memory/forward-tape memory (memory/valid-docs docs pad) q
                                                                      {:train? train? :rng rng
                                                                       :slot-docs docs :slot-mask pad})
                           result (decoder (assoc seq-in :memory output) train?)]
                       (assoc result :tape tape :output output)))
                   input-ids attention-mask labels doc-embeds doc-padding-mask question-embeds)
        total (reduce + 0 (map :n-targets rows))
        loss (/ (reduce + 0.0 (map #(* (:loss %) (:n-targets %)) rows)) (max total 1))]
    (when train?
      (doseq [{:keys [grad tape n-targets]} rows]
        (let [w (float (* scale (/ (double n-targets) (max total 1))))
              d (t/copy-floats (:data grad))]
          (dotimes [i (alength d)] (aset d i (float (* w (aget d i)))))
          (memory/backward! memory tape (t/matrix (:rows grad) (:cols grad) d)))))
    loss))

(defn train-step!
  "One optimizer step over collated batches (gradient accumulation: each
  batch's loss counts 1/count). Returns {:loss :lr :grad-norm :step}."
  [tr batches]
  (let [{:keys [memory optimizer schedule max-grad-norm]} tr
        n (count batches)]
    (memory/zero-grads! memory)
    (let [losses (mapv #(batch-loss! tr % {:train? true :scale (/ 1.0 n)}) batches)
          norm (optim/clip-grad-norm! memory max-grad-norm)
          lr (@schedule @(:step optimizer))
          step (optim/step! optimizer lr)]
      {:loss (util/mean losses) :lr lr :grad-norm norm :step step})))

(defn eval-loss
  "Mean loss of collated batches, without dropout or gradients."
  [tr batches]
  (util/mean (mapv #(batch-loss! tr % {:train? false}) batches)))

;; --- runtime learning ---------------------------------------------------------------------------------

(defn embed-example
  "An example {:question :answer :documents} as a training item, its
  documents and question embedded by enc."
  [enc {:keys [question answer documents]}]
  (let [texts (vec (concat (map str documents) [(str/trim (str question))]))
        vectors (encoder/embed-texts enc texts)
        n (count documents)]
    {"question" question "answer" answer
     :doc-embeds (t/take-rows vectors n)
     :question-embed (t/row vectors n)}))

(defn learn!
  "Tune the module on examples at runtime: each example a map
  {:question :answer :documents} (embedded with enc) or an item already
  carrying :doc-embeds and :question-embed. Runs :steps optimizer steps
  (default: one pass) over shuffled batches of :batch-size. Returns the
  step results."
  ([tr enc examples] (learn! tr enc examples {}))
  ([tr enc examples {:keys [steps batch-size] :or {batch-size 2}}]
   (let [items (mapv #(if (:doc-embeds %) % (embed-example enc %)) examples)
         batches-per-pass (int (Math/ceil (/ (count items) (double batch-size))))
         steps (or steps batches-per-pass)
         order (fn [] (sort-by (fn [_] (memory/uniform! (:rng tr))) (range (count items))))]
     (loop [k 0 queue [] results []]
       (if (= k steps)
         results
         (let [queue (if (seq queue) queue (vec (partition-all batch-size (order))))
               batch (collate tr (mapv items (first queue)) {})]
           (recur (inc k) (subvec queue 1) (conj results (train-step! tr [batch])))))))))

(defn snapshot
  "A copy of the module's weights and optimizer step, to roll back to."
  [tr]
  {:state (memory/state-dict (:memory tr))
   :step @(get-in tr [:optimizer :step])})

(defn restore!
  "Roll the module's weights back to a snapshot. The optimizer moments are
  kept (they only shape later updates)."
  [tr {:keys [state step]}]
  (doseq [[name {:keys [data]}] state] (memory/set-param! (:memory tr) name data))
  (reset! (get-in tr [:optimizer :step]) step)
  tr)

;; --- train.py: recipes ------------------------------------------------------------------------------

(def spec
  {:config {:desc "YAML recipe whose keys set option defaults."}
   :datasets {:desc "Dataset configs to train on (comma or space separated)." :coerce [:string]}
   :data_repo {:desc "Hub dataset repo or local directory." :default paths/data-repo}
   :data_dir {:desc "Local mirror of a Hub data repo." :default paths/data-dir}
   :system_prompt {:desc "System prompt of every example." :default data/default-system-prompt}
   :max_length {:desc "Token limit of a training example." :default 2048 :coerce :long}
   :decoder {:desc "Frozen decoder GGUF (Qwen3-8B)."}
   :encoder {:desc "Encoder GGUF, only for embeddings that must be computed."}
   :encoder_id {:desc "Encoder id naming the stored embeddings." :default paths/default-encoder}
   :num_memories {:desc "Memory tokens per example." :default memory/default-num-memories :coerce :long}
   :num_heads {:desc "Attention heads of the memory module." :default memory/default-num-heads :coerce :long}
   :num_self_attn_layers {:desc "Self-attention layers over the documents." :default memory/default-num-self-attn-layers :coerce :long}
   :num_cross_attn_layers {:desc "Refinement blocks." :default memory/default-num-cross-attn-layers :coerce :long}
   :dropout {:desc "Dropout of the memory module." :default memory/default-dropout :coerce :double}
   :retrieval_init {:desc "Start the slots from the documents most similar to the question." :default false}
   :init_memory {:desc "Checkpoint directory or weights file to start from."}
   :attn_implementation {:desc "Ignored (the decoder runs on ggml)." :default "flash_attention_2"}
   :learning_rate {:desc "Peak learning rate." :default 1e-4 :coerce :double}
   :weight_decay {:desc "AdamW weight decay." :default 0.01 :coerce :double}
   :warmup_ratio {:desc "Fraction of the optimiser steps spent on warmup." :default 0.1 :coerce :double}
   :batch_size {:desc "Examples per batch." :default 2 :coerce :long}
   :gradient_accumulation_steps {:desc "Batches per optimiser step." :default 8 :coerce :long}
   :max_epochs {:desc "Training epochs." :default 5 :coerce :long}
   :val_check_interval {:desc "Fraction of an epoch between validations." :default 1.0 :coerce :double}
   :seed {:desc "Random seed." :default util/default-seed :coerce :long}
   :devices {:desc "Ignored: one process, one device." :default 1 :coerce :long}
   :num_nodes {:desc "Ignored." :default 1 :coerce :long}
   :strategy {:desc "Ignored." :default "ddp"}
   :output_dir {:desc "Run directory (resumes if it exists)."}
   :num_workers {:desc "Ignored." :default 2 :coerce :long}
   :cache_dir {:desc "Ignored (models are GGUF paths)." :default paths/model-cache-dir}
   :wandb_project {:desc "Ignored."}
   :gpu {:desc "Put the decoder's training copy on the GPU when there is one." :default true}
   :f32_weights {:desc "Widen the decoder's matmul weights to f32 (keeps the backward pass on CUDA)." :default false}
   :n_threads {:desc "CPU threads for the decoder (0 = default)." :default 0 :coerce :long}})

(defn- coerce-value [k v]
  (let [c (get-in spec [k :coerce])]
    (cond
      (= c [:string]) (if (string? v) (vec (remove str/blank? (str/split v #"[,\s]+"))) (mapv str v))
      (= c :long) (if (string? v) (parse-long v) (long v))
      (= c :double) (if (string? v) (parse-double v) (double v))
      (#{:retrieval_init :gpu :f32_weights} k) (if (string? v) (util/str->bool v) (boolean v))
      :else v)))

(defn parse-args
  "Options from argv, with the defaults of --config applied first (explicit
  flags win). Unknown recipe keys and a missing datasets / output_dir are errors."
  [argv]
  (let [opts (cli/parse "usage: jolt train --config RECIPE.yaml --decoder Q.gguf [flags]"
                        (-> (update-vals spec #(select-keys % [:desc]))
                            (assoc-in [:datasets :coerce] []))
                        argv)
        recipe (when (:config opts) (util/read-recipe (:config opts)))
        unknown (sort (remove #(contains? spec (keyword %)) (keys recipe)))]
    (when (seq unknown)
      (throw (ex-info (str (:config opts) " sets unknown options: " (pr-str (vec unknown))) {:unknown unknown})))
    (let [defaults (into {} (for [[k {:keys [default]}] spec :when (some? default)] [k default]))
          from-recipe (into {} (for [[k v] recipe] [(keyword k) v]))
          merged (merge defaults from-recipe (into {} (remove (comp nil? val) opts)))
          merged (into {} (for [[k v] merged] [k (if (some? v) (coerce-value k v) v)]))]
      (when (empty? (:datasets merged))
        (throw (ex-info "--datasets is required (directly or through --config)." {})))
      (when-not (:output_dir merged)
        (throw (ex-info "--output_dir is required (directly or through --config)." {})))
      merged)))

;; --- train.py: the run ---------------------------------------------------------------------------------

(defn prepare-data
  "Fetch the data and embeddings of every config: {:train :validation :stores}."
  [opts make-encoder]
  (reduce
   (fn [acc config]
     (let [root (data/ensure-data config [data/train-split data/validation-split] (:data_repo opts) (:data_dir opts))
           splits (filterv #(seq (data/split-files config % root)) [data/train-split data/validation-split])]
       (when-not (some #{data/train-split} splits)
         (throw (ex-info (str "Config " (pr-str config) " has no training split in " (:data_repo opts) ".") {})))
       (emb/ensure-embeddings config splits {:repo (:data_repo opts) :data-dir (:data_dir opts)
                                             :encoder-id (:encoder_id opts) :make-encoder make-encoder})
       (reduce (fn [acc split]
                 (let [[dataset st] (emb/load-split-with-embeddings config split root
                                                                    [data/collection-column data/question-column
                                                                     data/answer-column]
                                                                    (:encoder_id opts))]
                   (-> acc
                       (assoc-in [:stores [config split]] st)
                       (update (if (= split data/train-split) :train :validation) conj [config split dataset]))))
               acc splits)))
   {:train [] :validation [] :stores {}}
   (:datasets opts)))

(defn- log-row! [output-dir row]
  (let [path (fs/path output-dir log-file)
        header "step,epoch,train_loss,val_loss,lr,grad_norm"]
    (fs/create-dirs (fs/parent path))
    (when-not (fs/exists? path) (spit (str path) (str header "\n")))
    (spit (str path) (str (str/join "," (map #(if (nil? %) "" %) ((juxt :step :epoch :train-loss :val-loss :lr :grad-norm) row)))
                          "\n")
          :append true)))

(defn save-resume!
  "Write the resume state to output-dir/last: weights, optimizer, progress."
  [tr output-dir progress]
  (let [dir (fs/path output-dir last-dir)]
    (memory/save (:memory tr) dir)
    (optim/save-state (:optimizer tr) (str (fs/path dir "optimizer.safetensors")))
    (util/write-json (fs/path dir "progress.json") progress)))

(defn load-resume!
  "Restore output-dir/last into the trainer; returns the progress map or nil."
  [tr output-dir]
  (let [dir (fs/path output-dir last-dir)]
    (when (fs/exists? (fs/path dir "progress.json"))
      (let [state (memory/load-memory-state (str dir))]
        (memory/load-partial! (:memory tr) state)
        (optim/load-state! (:optimizer tr) (str (fs/path dir "optimizer.safetensors")))
        (util/read-json (fs/path dir "progress.json"))))))

(defn fit!
  "train.py's loop: max-epochs over the shuffled training set in batches,
  gradient accumulation, validation every val-check-interval of an epoch,
  epoch-NN checkpoints, a resume state in last/ and logs/metrics.csv.
  Resuming restores weights, optimizer and step and skips the batches the
  interrupted epoch had done (in a fresh order, as Lightning does)."
  [tr {:keys [train validation stores batch-size accumulation max-epochs val-check-interval output-dir seed
              peak-lr warmup-ratio]}]
  (let [train-set (data/memory-dataset train)
        val-set (when (seq validation) (data/memory-dataset validation))
        n (data/dataset-size train-set)
        n-batches (int (Math/ceil (/ n (double batch-size))))
        steps-per-epoch (int (Math/ceil (/ n-batches (double accumulation))))
        total-steps (* steps-per-epoch max-epochs)
        val-every (when val-set (max 1 (int (* n-batches val-check-interval))))
        progress (or (load-resume! tr output-dir) {"epoch" 0 "batch" 0})
        validate (fn []
                   (let [m (data/dataset-size val-set)]
                     (eval-loss tr (for [start (range 0 m batch-size)]
                                     (collate tr (mapv #(data/dataset-item val-set %) (range start (min m (+ start batch-size))))
                                              stores)))))]
    (reset! (:schedule tr) (optim/lr-schedule {:peak peak-lr :total-steps total-steps :warmup-ratio warmup-ratio}))
    (fs/create-dirs output-dir)
    (println (str "training on " n " examples: " n-batches " batches an epoch, " total-steps " optimizer steps"))
    (doseq [epoch (range (get progress "epoch") max-epochs)]
      (let [order (vec (sort-by (fn [_] (memory/uniform! (:rng tr))) (range n)))
            batches (vec (partition-all batch-size order))
            skip (if (= epoch (get progress "epoch")) (get progress "batch") 0)]
        (loop [b skip pending []]
          (when (< b n-batches)
            (let [pending (conj pending (collate tr (mapv #(data/dataset-item train-set %) (nth batches b)) stores))
                  last-of-epoch? (= b (dec n-batches))
                  step? (or (= (count pending) accumulation) last-of-epoch?)
                  result (when step? (train-step! tr pending))
                  val-loss (when (and val-every (zero? (mod (inc b) val-every))) (validate))]
              (when (or result val-loss)
                (let [row {:step @(get-in tr [:optimizer :step]) :epoch epoch
                           :train-loss (:loss result) :val-loss val-loss :lr (:lr result) :grad-norm (:grad-norm result)}]
                  (log-row! output-dir row)
                  (println (str "epoch " epoch " batch " (inc b) "/" n-batches
                                (when result (format " loss %.4f lr %.2e" (:loss result) (:lr result)))
                                (when val-loss (format " val_loss %.4f" val-loss))))))
              (when val-loss
                (save-resume! tr output-dir {"epoch" epoch "batch" (inc b)}))
              (recur (inc b) (if step? [] pending)))))
        (memory/save (:memory tr) (fs/path output-dir (format epoch-dir-template epoch)))
        (save-resume! tr output-dir {"epoch" (inc epoch) "batch" 0})
        (println "saved" (str (fs/path output-dir (format epoch-dir-template epoch))))))
    tr))

(defn run
  "Train a memory module as train.py does."
  [opts]
  (when-not (:decoder opts) (throw (ex-info "--decoder (a GGUF file) is required" {})))
  (doseq [[k dflt] {:devices 1 :num_nodes 1}]
    (when (not= dflt (get opts k))
      (println (str "note: --" (name k) " " (get opts k) " is ignored: distilla trains in one process"))))
  (let [load-model (requiring-resolve 'distilla.llama/load-model)
        make-tokenizer (requiring-resolve 'distilla.llama/tokenizer)
        grad-model (requiring-resolve 'distilla.llama/grad-model)
        make-llama-encoder (requiring-resolve 'distilla.llama/encoder)
        _ ((requiring-resolve 'distilla.llama/quiet!) true)
        make-encoder (when (:encoder opts) #(make-llama-encoder (load-model (:encoder opts))))
        {:keys [train validation stores] :as prepared} (prepare-data opts make-encoder)
        embedding-dim (store/dim (val (first stores)))
        vocab (load-model (:decoder opts) {:vocab-only true})
        gm (grad-model (:decoder opts) {:gpu? (:gpu opts) :f32-weights (:f32_weights opts) :n-threads (:n_threads opts)})
        cfg (memory/config {:embedding-dim embedding-dim :output-dim (:n-embd gm)
                            :num-memories (:num_memories opts) :num-heads (:num_heads opts)
                            :num-self-attn-layers (:num_self_attn_layers opts)
                            :num-cross-attn-layers (:num_cross_attn_layers opts)
                            :dropout (:dropout opts) :retrieval-init (:retrieval_init opts)})
        module (memory/init-module cfg (:seed opts))
        output-dir (paths/resolve-path (:output_dir opts))]
    (when-let [init (:init_memory opts)]
      (let [{:keys [loaded missing unexpected]} (memory/load-partial! module (memory/load-memory-state (paths/resolve-path init)))]
        (println (str "Initialised memory module from " init " (" loaded " tensors)."))
        (when (seq missing) (println "Freshly initialised (absent from" init "):" missing))
        (when (seq unexpected) (println "Ignored (not part of this architecture):" unexpected))))
    (println (str "memory module: " (memory/parameter-count module) " parameters; decoder training copy on "
                  (if (:on-gpu? gm) "the GPU" "the CPU")))
    (try
      (fit! (trainer {:memory module :decoder (llama-decoder gm) :tokenizer (make-tokenizer vocab)
                      :system-prompt (:system_prompt opts) :max-length (:max_length opts)
                      :weight-decay (:weight_decay opts) :seed (:seed opts)})
            {:train train :validation validation :stores stores
             :batch-size (:batch_size opts) :accumulation (:gradient_accumulation_steps opts)
             :max-epochs (:max_epochs opts) :val-check-interval (:val_check_interval opts)
             :output-dir output-dir :seed (:seed opts)
             :peak-lr (:learning_rate opts) :warmup-ratio (:warmup_ratio opts)})
      (finally
        (.close ^java.io.Closeable gm)
        (.close ^java.io.Closeable vocab)
        (.close ^java.io.Closeable module)))))

(defn -main [& args]
  (cli/exit-on-error #(run (parse-args args))))
