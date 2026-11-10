(ns distilla.tokenizer
  "The tokenizer seam and the chat templates.

  The Python implementation adds <|memory|> to the decoder's vocabulary and
  overwrites the embeddings at those positions. A GGUF vocabulary can't be
  extended, and nothing needs it to be: the placeholder only marks where the
  memory vectors go. Tokenizers here split the text at <|memory|> (which is
  what a Hugging Face tokenizer does at an added special token), encode the
  pieces, and put memory-token-id at each placeholder. For llama.cpp that id
  is a sentinel that never reaches the model."
  (:require [clojure.string :as str]
            [distilla.util :as util]))

(defprotocol Tokenizer
  (encode-text [tok text] "Token ids of text containing no <|memory|> (special tokens parsed, no BOS/EOS added).")
  (decode-ids [tok ids] "Text of token ids, special tokens skipped.")
  (pad-id [tok] "Id used for left padding.")
  (memory-token-id [tok] "Id standing for <|memory|> in encoded prompts.")
  (chat-template [tok] "A fn [messages add-generation-prompt?] -> prompt text."))

(defn encode
  "Token ids of a prompt, <|memory|> mapped to memory-token-id."
  [tok text]
  (let [pieces (str/split text (re-pattern (java.util.regex.Pattern/quote util/memory-token)) -1)
        mem (memory-token-id tok)]
    (vec (mapcat identity
                 (interpose [mem] (map #(if (empty? %) [] (encode-text tok %)) pieces))))))

(defn decode [tok ids] (decode-ids tok ids))

(defn apply-chat-template
  "Render a chat with the tokenizer's template."
  [tok messages add-generation-prompt?]
  ((chat-template tok) messages add-generation-prompt?))

;; --- templates ----------------------------------------------------------------------

(defn chatml
  "Plain ChatML: <|im_start|>role\\ncontent<|im_end|>\\n per message."
  [messages add-generation-prompt?]
  (str (apply str (for [{:keys [role content]} messages]
                    (str "<|im_start|>" role "\n" content "<|im_end|>\n")))
       (when add-generation-prompt? "<|im_start|>assistant\n")))

(defn- strip-newlines [s] (-> s (str/replace #"^\n+" "") (str/replace #"\n+$" "")))

(defn qwen3
  "Qwen3's chat template with enable_thinking=False, for system/user/assistant
  turns without tools: the generation prompt carries an empty think block,
  and so does an assistant turn after the last user message."
  [messages add-generation-prompt?]
  (let [last-query (or (last (keep-indexed (fn [i m] (when (= "user" (:role m)) i)) messages)) -1)
        n (count messages)]
    (str
     (apply str
            (map-indexed
             (fn [i {:keys [role content]}]
               (if (= role "assistant")
                 (let [[reasoning content] (if (str/includes? content "</think>")
                                             (let [[before after] (str/split content #"</think>" 2)]
                                               [(strip-newlines (last (str/split before #"<think>")))
                                                (str/replace after #"^\n+" "")])
                                             ["" content])]
                   (if (and (> i last-query) (or (= i (dec n)) (seq reasoning)))
                     (str "<|im_start|>assistant\n<think>\n" reasoning "\n</think>\n\n"
                          (str/replace content #"^\n+" "") "<|im_end|>\n")
                     (str "<|im_start|>assistant\n" content "<|im_end|>\n")))
                 (str "<|im_start|>" role "\n" content "<|im_end|>\n")))
             messages))
     (when add-generation-prompt? "<|im_start|>assistant\n<think>\n\n</think>\n\n"))))
