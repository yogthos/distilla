"""Reference outputs of memorilla's rl/ scripts and the SkyRL-Memorilla TextWorld memory helpers, for the jolt port.

    MEMORILLA_SRC=... SKYRL_SRC=... python3 dev/make_rl_fixtures.py test/fixtures/rl/python.json
"""

import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, os.path.expanduser(os.environ.get("MEMORILLA_SRC", "~/src/memorilla/memorilla")))
sys.path.insert(0, os.path.join(os.environ["SKYRL_SRC"], "skyrl-gym", "skyrl_gym", "envs", "textworld"))

from rl.aggregate import average, score_rollouts  # noqa: E402
from rl.build_dataset import find_games, make_rows, split_games  # noqa: E402
from rl.generate_games import iter_specs, parse_mix  # noqa: E402

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "twmemory", os.path.join(os.environ["SKYRL_SRC"], "skyrl-gym", "skyrl_gym", "envs", "textworld", "memory.py"))
twmemory = importlib.util.module_from_spec(spec)
sys.modules["twmemory"] = twmemory
spec.loader.exec_module(twmemory)


def main(out: Path) -> None:
    result = {}
    with tempfile.TemporaryDirectory() as tmp:
        specs = iter_specs(parse_mix("medium:0.3,hard:0.4,long:0.2,extreme:0.1"), 42, Path(tmp))
        result["specs"] = [
            {"name": s.path.name, "tier": s.tier, "world_size": s.world_size, "nb_objects": s.nb_objects,
             "quest_length": s.quest_length, "quest_breadth": s.quest_breadth, "seed": s.seed}
            for s, _ in zip(specs, range(40))
        ]
        games_dir = Path(tmp) / "games"
        games_dir.mkdir()
        for s in result["specs"]:
            (games_dir / s["name"]).write_text("")
            if s["name"].endswith(".z8"):
                (games_dir / s["name"].replace(".z8", ".json")).write_text("")
        games = find_games([games_dir])
        splits = split_games(games, 3, 42)
        result["dataset"] = {}
        for split, members in splits.items():
            rows = make_rows(members)
            for row in rows:
                row["game_file"] = Path(row["game_file"]).name
            result["dataset"][split] = rows

    rollouts = []
    for i in range(30):
        tier = ["medium", "hard", "long"][i % 3]
        score = [0.0] * 5 + ([1.0] if i % 4 == 0 else [0.0])
        if i % 7 == 0:
            score = 0.5
        rollouts.append({"score": score, "stop_reason": ["stop", "length", "stop"][i % 3], "data_source": tier,
                         "env_extras": {"game_file": f"g{i // 3}", "extra_info": {"row_index": i // 6}}})
    result["rollouts"] = rollouts
    result["scores"] = score_rollouts(rollouts)
    result["average"] = average([score_rollouts(rollouts), score_rollouts(rollouts[:12])])

    responses = ["I think so.\n[ACTION: go north]", "<think>x</think>I will open the box.\nmore", "Let me take the key",
                 "[action:  unlock door with key ]", "", "\"look\".", "Action: eat apple; and then $$ (drop it)",
                 "x" * 300, "</think>", "I'll  go   west\n\nthen east"]
    result["actions"] = [[r, twmemory.parse_action(r)] for r in responses]

    memory = twmemory.TrajectoryMemory(turns_per_document=2, max_documents=3, max_observation_chars=15)
    for t in range(1, 8):
        memory.add_turn(turn=t, observation=f"Room {t}: a long observation text", action=f"go {t}",
                        reward=1.0 if t % 3 == 0 else 0.0, score=t // 3)
    memory.flush()
    result["documents"] = memory.documents
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "test/fixtures/rl/python.json"))
