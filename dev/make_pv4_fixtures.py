"""Run the Python PersonalizationV4 generator and builder on fixed inputs (a fake API), so the jolt port can be
checked against their exact outputs.

    MEMORILLA_SRC=~/src/memorilla/memorilla python3 dev/make_pv4_fixtures.py test/fixtures/pv4
"""

import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.expanduser(os.environ.get("MEMORILLA_SRC", "~/src/memorilla/memorilla")))

from personalizationv4 import prompts  # noqa: E402
from personalizationv4.build_dataset import build, parse_user_range  # noqa: E402
from personalizationv4.generate import GenerationConfig, UserGenerator  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

PERSONA = "Mara is a violin teacher in Lisbon who runs before dawn and keeps a secret recipe archive."
QUESTION_BLOCK = """<QUESTION>
{stem}
<CHOICE_A>
Option alpha for {stem}
<CHOICE_B>
Option bravo, with a comma, for {stem}
<CHOICE_C>
Option "charlie" for {stem}
<CHOICE_D>
Option delta for {stem}
<CHOICE_E>
Option echo for {stem}
<CORRECT_CHOICE>
{letter}
<RATIONALE>
Because of the persona.
<WORD_COUNTS>
A: 4 | B: 4 | C: 4 | D: 4 | E: 4
"""


class FakeResponses:
    def create(self, model, input, reasoning):
        prompt = input[0]["content"]
        if prompt.startswith(prompts.CATEGORIES_TEMPLATE[:40]):
            return SimpleNamespace(output_text="1. Teaching Style & Students\n2. Running Routine\n3. Food\n")
        if prompt.startswith(prompts.TOPICS_TEMPLATE[:40]):
            return SimpleNamespace(output_text="\n".join(f"{i + 1}. Topic number {i}." for i in range(12)))
        if prompt.startswith(prompts.CHAT_TEMPLATE[:40]):
            topic = prompt.split("2. **Scenario:**\n", 1)[1].split("\n", 1)[0]
            return SimpleNamespace(output_text=f"  Leo: Help with {topic}\nAssistant: Sure.  \n")
        category = prompt.split("following category:\n**", 1)[1].split("**", 1)[0]
        blocks = [QUESTION_BLOCK.format(stem=f"{category} question {i}?", letter="BDAEC"[i % 5]) for i in range(5)]
        return SimpleNamespace(output_text="\n".join(blocks) + "<QUESTION>\nA question without choices.\n")


def main(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    personas = root / "personas"
    personas.mkdir(parents=True)
    for user_id in (3, 4, 7):
        (personas / f"user_{user_id}.txt").write_text(PERSONA + f" ({user_id})", encoding="utf-8")
    config = GenerationConfig(num_categories=3, num_topics=12, num_questions=5, max_workers=4, max_retries=1)
    for user_id in (3, 4, 7):
        assert UserGenerator(SimpleNamespace(responses=FakeResponses()), config).run(
            personas / f"user_{user_id}.txt", root / "users" / f"user_{user_id}"
        )
    hard = root / "hard.csv"
    hard.write_text("user_id,question_index\n7,2\n7,9\n", encoding="utf-8")
    counts = build(root / "users", root / "data", parse_user_range("3-4"), parse_user_range("7"), 0.1, 23, hard,
                   "collection_id")
    expected = {split: pq.read_table(root / "data" / f"{split}.parquet").to_pylist() for split in counts}
    (root / "expected.json").write_text(json.dumps({"counts": counts, "rows": expected}))
    shutil.rmtree(root / "data")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "test/fixtures/pv4"))
