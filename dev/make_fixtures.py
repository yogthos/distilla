"""Generate test/fixtures from the Python reference implementation.

Needs torch, safetensors, nltk and rouge-score, plus a checkout of memorilla
(MEMORILLA_SRC, default ~/src/memorilla/memorilla) for its MemoryModule. The
memorilla modules are imported with their GPU-stack dependencies stubbed out,
so the fixture outputs come from the original forward pass.

    python3 dev/make_fixtures.py test/fixtures
"""

import json
import os
from pathlib import Path
import sys
import types

import torch


def stub_memorilla_deps() -> None:
    """Make memorilla.memory importable without transformers, huggingface_hub and yaml."""
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **kwargs: None
    transformers = types.ModuleType("transformers")
    transformers.PreTrainedTokenizerBase = object
    yaml = types.ModuleType("yaml")
    yaml.safe_load = lambda handle: {}
    for name, module in (("huggingface_hub", hub), ("transformers", transformers), ("yaml", yaml)):
        sys.modules.setdefault(name, module)
    sys.path.insert(0, os.path.expanduser(os.environ.get("MEMORILLA_SRC", "~/src/memorilla/memorilla")))


EMBEDDING_DIM = 16
OUTPUT_DIM = 12
NUM_MEMORIES = 4


def tolist(tensor: torch.Tensor) -> list:
    return tensor.detach().float().tolist()


def memory_fixtures(root: Path) -> None:
    from memorilla.memory import MemoryModule

    generator = torch.Generator().manual_seed(1)
    docs = torch.randn(2, 6, EMBEDDING_DIM, generator=generator)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[1, 3:] = True
    question = torch.randn(2, EMBEDDING_DIM, generator=generator)
    inputs = {"docs": tolist(docs), "mask": mask.tolist(), "question": tolist(question)}

    variants = {
        "default": dict(num_heads=4),
        "self_attn_retrieval": dict(num_heads=2, num_self_attn_layers=1, retrieval_init=True),
        "self2_cross1": dict(num_heads=4, num_self_attn_layers=2, num_cross_attn_layers=1),
        "inferred": dict(num_memories=7, num_cross_attn_layers=3),
    }
    summary = {"inputs": inputs, "variants": {}}
    for name, overrides in variants.items():
        options = dict(embedding_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM, num_memories=NUM_MEMORIES, dropout=0.0)
        options.update(overrides)
        torch.manual_seed(0)
        module = MemoryModule(**options).eval()
        directory = root / "memory" / name
        directory.mkdir(parents=True, exist_ok=True)
        if name == "self_attn_retrieval":
            module = module.to(torch.bfloat16)  # stored in bfloat16, compared in float32
            module.save(directory)
            module = module.float()
        elif name == "inferred":
            torch.save(module.state_dict(), directory / "memory.pt")  # no config.json: inferred from the weights
        else:
            module.save(directory)
        with torch.no_grad():
            masked = module(docs, mask, question)
            unmasked = module(docs, None, question)
            truncated = module(docs[1:, :3], None, question[1:])
            negated = module(docs, mask, -question)
        summary["variants"][name] = {
            "config": {**module.config.__dict__},
            "keys": sorted(module.state_dict().keys()),
            "masked": tolist(masked),
            "unmasked": tolist(unmasked),
            "truncated": tolist(truncated),
            "negated": tolist(negated),
        }

    # wrapped checkpoints of the default module
    torch.manual_seed(0)
    module = MemoryModule(EMBEDDING_DIM, OUTPUT_DIM, num_memories=NUM_MEMORIES, num_heads=4, dropout=0.0).eval()
    state = module.state_dict()
    wrapped = root / "memory" / "wrapped"
    wrapped.mkdir(parents=True, exist_ok=True)
    torch.save({f"module.{k}": v for k, v in state.items()}, wrapped / "module.pt")
    torch.save({f"memory.{k}": v for k, v in state.items()}, wrapped / "prefixed.pt")
    torch.save(
        {"state_dict": {**{f"memory.{k}": v for k, v in state.items()}, "llm.weight": torch.ones(1)}, "epoch": 3},
        wrapped / "lightning.pt",
    )
    # a non-contiguous tensor and assorted dtypes, for the unpickler
    misc = {
        "transposed": torch.arange(6, dtype=torch.float32).reshape(2, 3).t(),
        "sliced": torch.arange(10, dtype=torch.float32)[2:7],
        "half": torch.tensor([0.5, -1.25, 3.0], dtype=torch.float16),
        "long": torch.tensor([1, -2, 3000000000], dtype=torch.int64),
        "flag": True,
        "count": 7,
        "ratio": 0.25,
        "name": "x",
        "nested": {"list": [1, 2, 3], "tuple": (1, "a")},
    }
    torch.save(misc, wrapped / "misc.pt")

    with open(root / "memory" / "expected.json", "w") as handle:
        json.dump(summary, handle)

    gradient_fixtures(root, docs, mask, question)


def gradient_fixtures(root: Path, docs: torch.Tensor, mask: torch.Tensor, question: torch.Tensor) -> None:
    """Autograd gradients of sum(output * R) for every parameter, without dropout."""
    from safetensors.torch import save_file

    from memorilla.memory import MemoryModule

    variants = {
        "default": dict(num_heads=4),
        "retrieval_self1": dict(num_heads=2, num_self_attn_layers=1, retrieval_init=True),
        "self2_cross1": dict(num_heads=4, num_self_attn_layers=2, num_cross_attn_layers=1),
    }
    for name, overrides in variants.items():
        options = dict(embedding_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM, num_memories=NUM_MEMORIES, dropout=0.0)
        options.update(overrides)
        torch.manual_seed(0)
        module = MemoryModule(**options).train()  # dropout 0: train mode is deterministic
        directory = root / "grads" / name
        directory.mkdir(parents=True, exist_ok=True)
        module.save(directory)
        generator = torch.Generator().manual_seed(5)
        weights = torch.randn(2, module.num_memories, OUTPUT_DIM, generator=generator)
        output = module(docs, mask, question)
        (output * weights).sum().backward()
        tensors = {f"grad.{k}": p.grad.detach().clone() for k, p in module.named_parameters()}
        tensors["weights"] = weights
        tensors["output"] = output.detach()
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(directory / "grads.safetensors"))


def safetensors_fixtures(root: Path) -> None:
    from safetensors.torch import save_file

    tensors = {
        "f16": torch.tensor([[0.1, -2.5, 3.0, 65504.0], [1e-4, 0.0, -0.0, 7.0], [1.5, 2.5, -3.5, 4.5]], dtype=torch.float16),
        "bf16": torch.tensor([[1.0, -0.15625], [3.140625, 100.0]], dtype=torch.bfloat16),
        "f32": torch.tensor([[1.5, -2.25, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
        "i64": torch.tensor([3, 1, 4, 1 << 40], dtype=torch.int64),
        "i32": torch.tensor([7, -8, 9], dtype=torch.int32),
    }
    directory = root / "safetensors"
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "sample.safetensors"))
    expected = {name: {"dtype": str(t.dtype), "shape": list(t.shape), "data": t.float().flatten().tolist()}
                for name, t in tensors.items()}
    with open(directory / "expected.json", "w") as handle:
        json.dump(expected, handle)


class NoWordNet:
    """A WordNet with no synsets: METEOR then matches exactly and by stem only, as the jolt port does."""

    def synsets(self, word):
        return []


def metrics_fixtures(root: Path) -> None:
    from nltk.stem.porter import PorterStemmer
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    from nltk.translate.meteor_score import single_meteor_score
    from rouge_score import rouge_scorer

    words = (
        "caresses ponies ties caress cats feed agreed plastered bled motoring sing conflated troubled sized hopping "
        "tanned falling hissing fizzed failing filing happy sky skies dying lying tying news innings outings cannings "
        "howe proceed exceed succeed relational conditional rational valenci hesitanci digitizer conformabli radicalli "
        "differentli vileli analogousli vietnamization predication operator feudalism decisiveness hopefulness "
        "callousness formaliti sensitiviti sensibiliti triplicate formative formalize electriciti electrical hopeful "
        "goodness revival allowance inference airliner gyroscopic adjustable defensible irritant replacement adjustment "
        "dependent adoption homologou communism activate angulariti homologous effective bowdlerize probate rate cease "
        "controll roll generalizations oscillators running runs ran dogs dog studies studying study cried cries "
        "flies flying died agreement abilities ability fullness ought quickly quick quickest the a an by is was "
        "analogy logically biology apology meteorology ably anomaly ies eed ied dies hied spied tied "
        "university universal universe organization organs organic yelling yellowness sayings saying spying "
        "mississippi possessed possesses possession possessions xylophones yeses yy y ys oed ing"
    ).split()
    stemmer = PorterStemmer()
    stems = {word: stemmer.stem(word) for word in words}

    pairs = [
        ("the quick brown fox jumps over the lazy dog", "the quick brown fox jumps over the lazy dog"),
        ("the cat", "the cat sat on the mat"),
        ("running dogs", "the dog runs"),
        ("alpha beta", "gamma delta"),
        ("It is a guide to action which ensures that the military always obeys the commands of the party",
         "It is a guide to action that ensures that the military will forever heed Party commands"),
        ("Paris is the capital of France.", "The capital of France is Paris"),
        ("He studied the organizations carefully", "she was studying organisational structures carefully"),
        ("", "something"),
        ("yes. the results were significant", "Yes. Results were statistically significant in all trials."),
        ("a a b", "a c"),
    ]
    predictions = [p for p, _ in pairs]
    references = [r for _, r in pairs]

    def tok(text):
        return text.lower().split()

    smoothing = SmoothingFunction().method1
    bleu_pairs = []
    for p, r in pairs:
        bleu_pairs.append({
            "bleu1": corpus_bleu([[tok(r)]], [tok(p)], weights=(1, 0, 0, 0), smoothing_function=smoothing),
            "bleu4": corpus_bleu([[tok(r)]], [tok(p)], weights=(0.25,) * 4, smoothing_function=smoothing),
        })
    corpus = {
        "bleu1": corpus_bleu([[tok(r)] for r in references], [tok(p) for p in predictions],
                             weights=(1, 0, 0, 0), smoothing_function=smoothing),
        "bleu4": corpus_bleu([[tok(r)] for r in references], [tok(p) for p in predictions],
                             weights=(0.25,) * 4, smoothing_function=smoothing),
    }
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge = []
    for p, r in pairs:
        score = scorer.score(r, p.strip())
        rouge.append({"rouge1": score["rouge1"].fmeasure, "rougeL": score["rougeL"].fmeasure})
    meteor = [single_meteor_score(tok(r), tok(p), wordnet=NoWordNet()) for p, r in pairs]

    directory = root
    with open(directory / "metrics.json", "w") as handle:
        json.dump({"stems": stems, "pairs": pairs, "bleu": bleu_pairs, "corpus_bleu": corpus,
                   "rouge": rouge, "meteor": meteor}, handle, indent=1)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "test/fixtures")
    root.mkdir(parents=True, exist_ok=True)
    stub_memorilla_deps()
    memory_fixtures(root)
    safetensors_fixtures(root)
    metrics_fixtures(root)


if __name__ == "__main__":
    main()
