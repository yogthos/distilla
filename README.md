# distilla

A [Jolt](https://jolt-lang.net) port of [Memorilla](https://github.com/snap-stanford/memorilla)
(*Latent Semantic Memory for LLMs*). Memorilla compresses a whole document collection into a
fixed number of question-dependent memory tokens (16 by default) for a frozen LLM. This port
**trains** the memory module against the frozen decoder, both as `train.py`'s recipe runner and as
a runtime API for tuning a module on the fly. It also **runs** it: it loads the published checkpoints
and data, generates with the frozen decoder, and scores the benchmarks. It needs no Python, PyTorch
or vLLM.

| Python (memorilla)                     | Jolt (distilla)                                         | Backed by                       |
| -------------------------------------- | ------------------------------------------------------- | ------------------------------- |
| `memory.py` `MemoryModule`             | `distilla.memory`                                       | CBLAS (OpenBLAS / Accelerate)   |
| `torch.load` of `memory.pt`            | `distilla.torch` (zip + pickle reader)                  | pure Clojure                    |
| `safetensors`                          | `distilla.safetensors`                                  | libc stdio                      |
| `embeddings.py` store / writer         | `distilla.store`, `distilla.embeddings`                 |                                 |
| `data.py` (datasets, parquet)          | `distilla.data`, `distilla.parquet`                     | DuckDB C API                    |
| `vllm.py` `MemoryVLLM`, `llm.py`       | `distilla.model`, `distilla.llama`                      | llama.cpp through a C shim      |
| `metrics.py` (nltk, rouge_score)       | `distilla.metrics`, `distilla.porter`                   | pure Clojure                    |
| `train.py`, `llm.py` (training)        | `jolt train`, `distilla.train`, `distilla.optim`        | ggml autodiff + BLAS + C kernels|
| `personalizationv4/`                   | `jolt pv4-generate`, `jolt pv4-build`                   | any OpenAI-compatible API       |
| `rl/` + SkyRL-Memorilla (GRPO, envs)   | `jolt rl-*`, `distilla.rl.*`                            | TextWorld simulator in Clojure  |
| `benchmarks.py`, `paths.py`, `utils.py`| `distilla.benchmarks`, `distilla.paths`, `distilla.util`|                                 |
| `evaluate.py`                          | `jolt evaluate` (`distilla.evaluate`)                   |                                 |
| `evaluate_baselines.py`                | `jolt baselines` (`distilla.baselines`)                 |                                 |
| README "getting started" example       | `jolt answer` (`distilla.answer`)                       |                                 |

Everything in the Python repository is ported, including the parts of SkyRL-Memorilla that
`rl/` runs on: the TextWorld environments, the trajectory memory, the multi-turn rollout loop and
GRPO. Two things stay with TextWorld itself, as they do in the Python version. Game *generation*
calls TextWorld's generator (`tw-make` and `textworld.generator`). Compiled `.z8` games play in its
Inform7 interpreter, through `tools/textworld_bridge.py`. Spec games (`.json`) run in a Clojure port
of the fork's in-process simulator. Multi-GPU data parallelism (`--devices`, `--strategy`) is
accepted and ignored: training runs in one process. Checkpoints trained
by the Python code load as they are: `memory.pt` (state dict, Lightning-style `state_dict` wrapper,
`memory.`/`module.`/`_orig_mod.` prefixes) or `memory.safetensors`, with or without `config.json`.

## Setup

You need [jolt](https://jolt-lang.net) 0.8.25 or later, a C compiler, CMake, git, curl and unzip, and
OpenBLAS on Linux (`libopenblas0` / `openblas`). macOS uses the Accelerate framework.

```bash
jolt native                    # DuckDB (prebuilt) + llama.cpp + native/libdistilla_llama, CPU build
DISTILLA_GPU=cuda jolt native  # or build llama.cpp with CUDA (also: metal, vulkan, hip)
jolt test                      # unit tests: no GPU, no models, no network
```

`DISTILLA_SKIP_LLAMA=1 jolt native` fetches only DuckDB, which is all the unit tests need.
To rebuild llama.cpp with another backend, delete `native/llama.cpp/build-*` and
`native/libdistilla_llama.*` first.

### Models

The decoder and the encoder run from GGUF files. The released checkpoints were trained with
**Qwen3-8B** (decoder, hidden size 4096) and **Qwen3-Embedding-4B** (encoder, 2560), so use GGUF
conversions of exactly those. Prefer BF16/F16 or Q8_0 for results close to the paper; lower-bit
quantisations work but shift the numbers. For example:

```bash
hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q8_0.gguf --local-dir models
hf download Qwen/Qwen3-Embedding-4B-GGUF Qwen3-Embedding-4B-Q8_0.gguf --local-dir models
```

The memory checkpoints come from `memorilla/Memorilla-Qwen3-8B` on the Hub (downloaded on first use
when `--checkpoint` is a repo id, or fetch them with `hf download memorilla/Memorilla-Qwen3-8B --local-dir checkpoints`).

## Usage

```bash
# One question from a dataset row or your own documents (the README example of the Python repo).
jolt answer --checkpoint memorilla/Memorilla-Qwen3-8B --subfolder pmv2 \
  --decoder models/Qwen3-8B-Q8_0.gguf --encoder models/Qwen3-Embedding-4B-Q8_0.gguf \
  --config pmv2 --row 73

# Evaluate a checkpoint on every benchmark, or one (data and embeddings download on first use).
jolt evaluate --benchmark all --checkpoint checkpoints/stage3 \
  --decoder models/Qwen3-8B-Q8_0.gguf --encoder models/Qwen3-Embedding-4B-Q8_0.gguf
jolt evaluate --benchmark pv4 --checkpoint checkpoints/pv4 --decoder ... --encoder ...

# Your own data: data/<config>/<split>-00000-of-00001.parquet under a local repo directory.
jolt evaluate --data_repo my_data --config my_task --split test --scoring generation \
  --checkpoint runs/my_task --decoder ... --encoder ...

# Text-only baselines.
jolt baselines --method closed_book --benchmark all --decoder ... --encoder ...
jolt baselines --method rag --top_k 5 --benchmark all --decoder ... --encoder ...
jolt baselines --method full_context --benchmark triviaqa --decoder ... --encoder ...
```

Every entry point takes `--help`. Flags keep the Python spellings (`--max_length`, `--data_repo`, ...).
`--encoder` is needed for choice scoring (pmv2, pv4, factkg) and for computing embeddings the data
repository does not publish; the published ones are read from
`embeddings/qwen3-embedding-4b/...` (`--encoder_id` names that folder). `--n_gpu_layers` controls
offloading (default: all layers).

Results are laid out as in the Python version: `results/<benchmark>/predictions.jsonl` and
`metrics.json` (percentages), `results/baselines/<method>/<benchmark>/context.json` for baselines.

Environment: `DISTILLA_HOME`, `DISTILLA_DATA_DIR`, `DISTILLA_DATA_REPO`, `DISTILLA_MODEL_CACHE`
(the Python version's `MEMORILLA_*`), plus `HF_TOKEN`, `HF_ENDPOINT` and `HF_HUB_OFFLINE` for Hub
downloads, which go through `curl`.

## Training

```bash
# train.py's recipes, unchanged (data and embeddings download on first use)
jolt train --config configs/stage1_enwiki.yaml --decoder models/Qwen3-8B-BF16.gguf
jolt train --config configs/single_task.yaml --datasets lamp7 --output_dir runs/lamp7 \
  --init_memory checkpoints/stage2 --decoder models/Qwen3-8B-BF16.gguf
```

Flags and recipe keys are `train.py`'s, with the same precedence (flags win; an unknown recipe key is
an error). A run writes `epoch-NN/` checkpoints (`memory.safetensors` + `config.json`, loadable by
`evaluate` and `answer`), a resume state in `last/` (weights, AdamW moments, progress), and
`logs/metrics.csv`. Rerunning the same command resumes. Extra flags: `--gpu` (put the decoder's
training copy on the GPU, default true), `--f32_weights` (see below) and `--n_threads`.

### Tuning at runtime

```clojure
(require '[distilla.llama :as llama] '[distilla.memory :as memory] '[distilla.train :as train])

(def dec    (llama/load-model "Qwen3-8B-BF16.gguf" {:vocab-only true}))   ; tokenizer
(def grad   (llama/grad-model "Qwen3-8B-BF16.gguf"))                      ; frozen decoder, backward pass
(def enc    (llama/encoder (llama/load-model "Qwen3-Embedding-4B-Q8_0.gguf")))
(def module (memory/from-pretrained "checkpoints/stage3"))

(def tr (train/trainer {:memory module :decoder (train/llama-decoder grad)
                        :tokenizer (llama/tokenizer dec) :learning-rate 1e-5}))

(def before (train/snapshot tr))
(train/learn! tr enc [{:question "..." :answer "..." :documents ["..." "..."]}] {:steps 10})
;; => [{:loss 2.31 :lr 1.0E-5 :grad-norm 0.8 :step 1} ...]
(train/restore! tr before)          ; roll the update back, or keep it
(memory/save module "runs/tuned")   ; or keep serving it: the model namespace reads it live
```

`learn!` embeds the examples' documents and questions with the encoder, collates them as training
does, and runs AdamW steps (constant learning rate unless `:schedule` is given). `train-step!` takes
collated batches directly, and `eval-loss` scores them without gradients. A `distilla.model`
built on the same module generates with the updated weights immediately.

### How training works

The loss is the decoder's next-token cross-entropy over the answer tokens, with the memory tokens in
the system turn, exactly as in `llm.py`. Only the memory module learns:

1. `distilla.memory/forward-tape` runs the module and keeps what the backward pass needs.
2. `native/distilla_train.c` runs the frozen decoder as a ggml graph built from the GGUF weights
   (Llama / Qwen2 / Qwen3 layouts), with the memory vectors as its only parameter. ggml's autodiff
   returns the loss and d(loss)/d(memory tokens). llama.cpp's public API trains model tensors, not
   input embeddings, which is why this graph exists. Tests check it against llama.cpp's inference
   (memory segments included) and against finite differences.
3. `distilla.memory/backward!` turns that gradient into parameter gradients: a hand-written backward
   for attention, LayerNorm, GELU, dropout and linear layers on BLAS, checked against PyTorch
   autograd (`test/fixtures/grads`).
4. `distilla.optim` clips the global gradient norm at 1, applies torch's AdamW (betas 0.9/0.999,
   weight decay 0.01) through `native/distilla_ops.c`, and follows `train.py`'s schedule: linear
   warmup from 1e-8 of the peak over `warmup_ratio` of the steps, then cosine decay to 10%.

The decoder's training copy is loaded separately from the inference context, so training and
serving the same 8B model at once holds its weights twice. ggml computes the gradient through a
weight matrix with `out_prod`, which CUDA supports only for F32 weights. With BF16/F16/quantised
weights, ggml's scheduler runs those products on the CPU. That is correct but slow for an 8B model.
`--f32_weights` widens the weights so the backward pass stays on the GPU, at 4 bytes per weight
(~32 GB for Qwen3-8B). Metal and CPU-only builds have no such restriction.

## Long sequences and prefix reuse

**Decoding.** A decoder remembers what its KV cache holds and decodes only what follows the longest
common prefix of the new prompt (`:reuse-prefix`, on by default). Memorilla prompts share their
system-prompt prefix. A multi-turn agent's prompt extends the previous turn, so its history is never
re-decoded (unless memory vectors before it changed). llama.cpp's results depend slightly on batch
shape, so a reused cache agrees with a fresh decode to about 1e-3 relative, not bit for bit. The
tests check that bound, including after a generated continuation.

**Training.** Sequences up to 1024 positions run through one autodiff graph. Longer ones (RL
trajectories reach 16k tokens) use a layer-wise backward pass:

- The forward pass keeps only each layer's input on the host.
- The backward pass walks down one layer at a time, recomputing it for one chunk of query rows and
  pulling the upstream gradient through it.
- Key and value gradients are summed over chunks, then pushed through the K/V projections.

Peak memory is heads × chunk × length attention floats instead of layers × heads × length². Tests
show it matches the single-graph gradients. `distilla.llama/set-chunking!` tunes the threshold and
chunk size.

## PersonalizationV4

```bash
# Rebuild the released parquet files from the released raw outputs (no API calls)
hf download memorilla/PersonalizationV4 --repo-type dataset --include "raw/*" --local-dir pv4_data
jolt pv4-build --users_dir pv4_data/raw/users --hard_subset pv4_data/raw/hard_subset.csv --output_dir pv4_data/data

# Generate new users (OpenAI Responses API, or --api chat --base_url http://localhost:8080/v1 for llama-server)
OPENAI_API_KEY=... jolt pv4-generate --persona_dir pv4_data/raw/personas --output_dir pv4_data/users
jolt pv4-build --users_dir pv4_data/users --output_dir pv4_data/new --id_column collection_id
```

The pipeline's two seeded steps use bit-exact ports of the generators Python used. Answer
positions use `random.Random` (Mersenne Twister); the per-user train/validation split uses numpy's
`default_rng(23).permutation`. On identical model outputs, the question tables and parquet files
come out byte-identical to the Python pipeline's (`test/fixtures/pv4`). Generation is resumable, as
in Python.

## Reinforcement learning on TextWorld

```bash
pip install textworld==1.7.0     # game generation and the Inform7 interpreter (as for the Python rl/)
jolt rl-games --output_dir games/medium_hard --tiers medium:0.3,hard:0.7 --num_games 400 --workers 16
for tier in long mega huge extreme; do jolt rl-games --output_dir games/$tier --tiers $tier:1 --num_games 20; done
jolt rl-dataset --games_dirs games/* --output_dir data/textworld

jolt rl-train --config configs/rl/textworld_grpo.yaml \
  --train_data data/textworld/train.parquet --val_data data/textworld/validation.parquet \
  --decoder models/Qwen3-4B-Instruct-2507-BF16.gguf --encoder models/Qwen3-Embedding-4B-BF16.gguf \
  --ckpt_path runs/textworld/checkpoints --export_path runs/textworld/exports

jolt rl-eval --config configs/rl/textworld_eval.yaml --val_data data/textworld/test.parquet \
  --decoder ... --encoder ... --model runs/textworld/checkpoints/global_step_30 --export_path evals/memorilla
jolt rl-eval ... --model untrained --export_path evals/untrained    # or --model no-memory
jolt rl-aggregate evals/untrained/seed* --output evals/untrained/metrics.json
jolt rl-export runs/textworld/checkpoints/global_step_30 runs/textworld/memory   # also reads SkyRL checkpoints
```

The settings follow `rl/configs/textworld_grpo.yaml`:

- **Episodes:** K = 8 memory vectors, recomputed every turn from the trajectory documents with the
  mean document embedding as the query.
- **Rollouts:** 16 per prompt at temperature 1.0, up to 128 tokens a turn and 50 turns.
- **GRPO:** group-normalised advantages, token-mean loss over micro-batches of 2, no KL term.
- **Optimisation:** AdamW at a constant 5e-4, gradient clipping at 1, 4 prompts per step,
  validation and a checkpoint every 3 steps.

Rollouts follow SkyRL's token-in-token-out layout: replies with their stop token at loss mask 1,
observations as chat-template tokens at mask 0. As in SkyRL, the training pass scores each
trajectory as one sequence, with memory computed from its final documents.

Game parameters, dataset splits and prompt variants match the Python scripts exactly (ported seeded
streams). The aggregate metrics, action parsing and memory documents are checked against the Python
code (`test/fixtures/rl`).

## How it works

The memory module is a fixed-shape transformer head: learned (or retrieved) slots attend to the
question, then to the document embeddings, and are projected to the decoder's width. Its weight
matrices live in foreign memory and every product is a `cblas_sgemm`. Element-wise work (layer norm,
exact GELU, softmax) is in Clojure on float arrays.

The Python version adds `<|memory|>` to the decoder vocabulary and overwrites those positions'
input embeddings. A GGUF vocabulary can't be extended, and it doesn't need to be. The prompt is split
at the placeholders into segments of token ids and memory vectors, and `native/distilla_llama.c`
decodes them in order into one KV cache. llama.cpp passes its parameter structs by value and changes
them between releases, so the shim gives jolt a small scalar-and-pointer API instead. It is pinned to
llama.cpp b12239.

## Differences from the Python version

- **Precision.** The memory module runs and trains in float32 (Python: bf16-mixed on GPU). The
  decoder and encoder precision is whatever the GGUF files hold. Training's decoder graph computes
  in float32.
- **Training randomness.** Initialisation and dropout follow torch's distributions, not its random
  streams, so a run from scratch is not bit-identical to the Python one.
- **METEOR** has nltk's exact and Porter-stem matching stages but not its WordNet synonym stage, so
  it can come out slightly lower than nltk's when a prediction uses a synonym. BLEU (corpus, method-1
  smoothing), ROUGE-1/L (rouge_score with the Porter stemmer), token F1, exact match and polarity
  reproduce the Python numbers. Fixtures from nltk/rouge_score check this.
- **Encoder.** Qwen3-Embedding runs with last-token pooling and L2 normalisation. An EOS token is
  appended to every text, as its Hugging Face tokenizer does (`:append-eos` in `distilla.llama/encoder`).
- **Chat template.** Qwen3's template with thinking disabled is built in (`distilla.tokenizer/qwen3`),
  not read from the model. Prompts are truncated on the right to `--max_length`, as the Python
  collator does.
- **retrieval_init with padding.** In a padded batch, PyTorch's top-k runs over the padded width, so
  a padding row can fill a slot when a collection has fewer documents than slots. The batched path
  reproduces this, so `--batch_size` affects factkg the same way it does in Python.
- No cross-process file lock around embedding downloads, and no subprocess isolation for the encoder
  (vLLM needed it; llama.cpp doesn't).
- **RL rollouts** run one at a time on one llama.cpp context. SkyRL batches them in vLLM; the
  results are the same, but a step takes longer.
- **RL reward placement.** SkyRL can misplace the first turn's reward in the token-level reward list
  when memory placeholders are inserted later. Returns (sums) are unaffected and so is GRPO; here
  each reward sits at the end of its own reply.
- Training checkpoints are `memory.safetensors`, not `memory.pt`. Both load everywhere here. The
  Python code reads safetensors with `safetensors.torch.load_file` if you need to go back.

## Tests

`jolt test` runs the ported unit tests: registry, recipes, data access and collation, the embedding
store layout, metrics, the memory module and the model wiring with a fake decoder, and training. The
training tests cover the backward pass, AdamW, clipping, the schedule, recipes, the training loop
with a fake decoder, checkpoints and resume. The memory module
and the file readers are checked against fixtures that `dev/make_fixtures.py` produces with the
original PyTorch `MemoryModule`, `torch.save`, `safetensors`, nltk and rouge_score. The forward pass
agrees to about 1e-6. To regenerate them:

```bash
python3 -m venv .venv && .venv/bin/pip install torch safetensors nltk rouge-score numpy
MEMORILLA_SRC=~/src/memorilla/memorilla .venv/bin/python dev/make_fixtures.py test/fixtures
```

Optional suites:

- `DISTILLA_TEST_GGUF=path/to/any-small.gguf jolt test` exercises the llama.cpp shim: tokenisation,
  segmented decoding and pooled embeddings. It also checks the training graph against llama.cpp's
  inference and finite differences, and that runtime learning lowers the loss.
  `stories260K.gguf` from `ggml-org/models` (1 MB) is enough. `dev/make_qwen3_toy.py` rewrites it as
  a Qwen3-architecture model (QK-norm, NEOX RoPE) to cover that layout.
- `DISTILLA_TEST_CHECKPOINTS=checkpoints jolt test` loads every `memory.pt` below that directory
  strictly and runs a forward pass.

## License

Apache 2.0, as the original.
