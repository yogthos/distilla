"""Generate small TextWorld games and play scripted command sequences through the Python
FastTextWorldSimulator, so the jolt port (distilla.rl.sim) can be checked against its exact outputs.

    FAST_SIM_DIR=.../skyrl-gym/skyrl_gym/envs/textworld PYTHONHASHSEED=0 \
        python3 dev/make_rl_sim_fixtures.py test/fixtures/rl

Needs textworld (only to generate the games) and fast_sim.py (stdlib only).

Writes, per game, `<name>.json` (a game spec trimmed to the keys the simulator reads) and
`<name>.transcript.json`:

    {"game": "<name>.json",
     "observations": [...],  # every distinct observation string, referenced by index
     "privileged": [...],    # every distinct get_privileged_state() result, referenced by index
     "episodes": [{"reset": {"obs": i, "info": {...}, "privileged": j},
                   "steps": [{"cmd": ..., "obs": i, "reward": ..., "done": ..., "info": {...},
                              "privileged": j  # on some steps only
                             }, ...]}, ...]}

The one hash-order dependence in fast_sim.py: `_resolve_target_in_inventory` matches against
`list(self._inventory)`, a set of strings, so when two carried items score the same for a query the
winner depends on PYTHONHASHSEED. The script patches that method to use `sorted(self._inventory)`
(the port does the same) and runs everything under two hash seeds to check the transcripts agree.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

FAST_SIM_DIR = os.environ.get(
    "FAST_SIM_DIR",
    "/tmp/claude-1000/-home-yogthos-src-distilla/0e4c7a47-cda2-468b-b3f6-c014236cf26d/scratchpad/"
    "skyrl/skyrl-gym/skyrl_gym/envs/textworld",
)
sys.path.insert(0, FAST_SIM_DIR)

from fast_sim import FastTextWorldSimulator  # noqa: E402


def _resolve_target_in_inventory_sorted(self, name):
    name_lower = name.lower().strip()
    if not name_lower:
        return None
    return self._match_entity(name_lower, sorted(self._inventory))


FastTextWorldSimulator._resolve_target_in_inventory = _resolve_target_in_inventory_sorted

# name, seed, rooms, objects, quest length, quest breadth, extra fail event (food id to forbid)
GAMES = [
    ("locked_doors", 101, 6, 10, 5, 1, None),
    ("two_quests", 103, 6, 10, 5, 2, None),
    ("supporters", 22, 6, 10, 5, 2, None),
    ("fail_event", 12, 5, 10, 4, 1, "auto"),
]


def trim_conditions(events):
    return [{"condition": {"preconditions": e["condition"]["preconditions"]}} for e in events]


def make_spec(seed, rooms, objects, length, breadth):
    import textworld
    from textworld.generator import make_game

    o = textworld.GameOptions()
    o.seeds = seed
    o.nb_rooms = rooms
    o.nb_objects = objects
    o.quest_length = length
    o.quest_breadth = breadth
    d = json.loads(json.dumps(make_game(o).serialize()))
    infos = [
        [eid, {k: info.get(k) for k in ("type", "name", "noun", "adj", "desc")}] for eid, info in d["infos"]
    ]
    quests = [
        {
            "desc": q.get("desc"),
            "reward": q.get("reward", 1),
            "commands": q.get("commands", []),
            "win_events": trim_conditions(q.get("win_events", [])),
            "fail_events": trim_conditions(q.get("fail_events", [])),
        }
        for q in d["quests"]
    ]
    return {"objective": d.get("objective", ""), "world": d["world"], "quests": quests, "infos": infos}


def add_fail_event(spec):
    """Losing condition: eating a food item the walkthrough never eats."""
    walk = " ".join(c for q in spec["quests"] for c in q["commands"])
    foods = [eid for eid, info in spec["infos"] if info["type"] == "f" and info["name"] not in walk]
    if not foods:
        return None
    food = foods[0]
    spec["quests"][0]["fail_events"].append(
        {"condition": {"preconditions": [{"name": "eaten", "arguments": [{"name": food, "type": "f"}]}]}}
    )
    return food


def entity_names(spec):
    names = {"o": [], "f": [], "k": [], "c": [], "s": [], "d": []}
    for eid, info in spec["infos"]:
        if info["type"] in names and info["name"]:
            names[info["type"]].append(info["name"])
    return names


def probe_commands(spec):
    """Edge cases that every game should answer."""
    n = entity_names(spec)
    portable = n["o"] + n["f"] + n["k"]
    first = lambda xs, default: xs[0] if xs else default  # noqa: E731
    obj, key, cont, sup, door = (
        first(portable, "thing"),
        first(n["k"], "key"),
        first(n["c"], "box"),
        first(n["s"], "table"),
        first(n["d"], "door"),
    )
    food = first(n["f"], "apple")
    return [
        "", "   ", "look", "l", "LOOK", "i", "inventory", "xyzzy", "jump around", "examine", "take", "x",
        "examine unicorn", "x unicorn", "take unicorn", "drop unicorn", "open unicorn", "close unicorn",
        "eat unicorn", "go up", "n", "s", "e", "w", "north", "go south", "GO EAST", "west",
        "unlock", "unlock " + door, "unlock " + door + " with unicorn", "unlock unicorn with " + key,
        "lock " + cont, "lock " + cont + " with " + key, "lock " + door + " with " + key,
        "put " + obj, "put " + obj + " on unicorn", "put unicorn on " + sup,
        "insert " + obj, "insert " + obj + " into " + cont, "insert unicorn into " + cont,
        "take " + obj + " from unicorn", "take unicorn from " + cont, "take unicorn from " + sup,
        "take " + obj + " FROM " + sup, "eat " + obj, "eat " + food, "drop " + obj,
        "examine " + door, "open " + door, "close " + door, "x " + cont, "examine the " + sup,
        "open " + cont, "close " + cont, "take " + cont, "take " + sup, "take " + door, "eat " + cont,
        "examine P", "examine I", "examine r_0", "x c_0", "take k_0", "examine " + obj.upper(),
        "take a " + obj, "take an " + obj, "examine " + obj[:3], "inventory",
    ]


def random_command(spec, sim, rng):
    """A plausible command. Names come mostly from what is in reach (read from the simulator's
    state; only the command text goes in the transcript), sometimes from anywhere in the game."""
    n = entity_names(spec)
    nearby = set(sim._inventory)
    nearby.update(e for e, loc in sim._obj_locations.items() if loc == sim._player_room)
    nearby.update(e for e, c in sim._obj_in_container.items() if sim._obj_locations.get(c) == sim._player_room)
    nearby.update(e for e, c in sim._obj_on_supporter.items() if sim._obj_locations.get(c) == sim._player_room)
    nearby.update(d for (a, b), d in sim._door_links.items() if sim._player_room in (a, b))

    def pool(*types):
        near = sorted(sim._get_name(e) for e in nearby if sim._get_type(e) in types and sim._infos[e].get("name"))
        anywhere = [x for t in types for x in n[t]]
        if near and rng.random() < 0.8:
            return near
        return anywhere or near or ["unicorn"]

    def name(*types):
        x = rng.choice(pool(*types))
        r = rng.random()
        if r < 0.1:
            return "the " + x
        if r < 0.15:
            return x.split()[-1]
        if r < 0.18:
            return x.upper()
        if r < 0.2:
            return x[: max(1, len(x) // 2)]
        return x

    held = "okf"
    makers = [
        (6, lambda: rng.choice(["go north", "go south", "go east", "go west", "n", "s", "e", "w"])),
        (1, lambda: rng.choice(["look", "inventory", "i", "l"])),
        (3, lambda: "examine " + name("o", "f", "k", "c", "s", "d")),
        (4, lambda: "take " + name("o", "f", "k")),
        (1, lambda: "take " + name("c", "s", "d")),
        (2, lambda: "take " + name("o", "f", "k") + " from " + name("c", "s")),
        (2, lambda: "drop " + name(*held)),
        (3, lambda: "open " + name("c", "d")),
        (1, lambda: "close " + name("c", "d")),
        (3, lambda: "unlock " + name("c", "d") + " with " + name("k")),
        (1, lambda: "unlock " + name("c", "d") + " with " + name("o", "f")),
        (1, lambda: "lock " + name("c", "d") + " with " + name("k")),
        (2, lambda: "put " + name(*held) + " on " + name("s", "c")),
        (2, lambda: "insert " + name(*held) + " into " + name("c", "s")),
        (2, lambda: "eat " + name("f", "o")),
        (1, lambda: rng.choice(["dance", "examine", "unlock it", "put it", "insert it", "x nothing"])),
    ]
    weights = [w for w, _ in makers]
    return rng.choices(makers, weights)[0][1]()


def walkthrough(spec):
    return [c for q in spec["quests"] for c in q["commands"]]


class Table:
    """Interns values so a transcript stores each distinct one once."""

    def __init__(self):
        self.values, self.index = [], {}

    def __call__(self, value):
        k = json.dumps(value, sort_keys=True)
        if k not in self.index:
            self.index[k] = len(self.values)
            self.values.append(value)
        return self.index[k]


def run(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    rng = random.Random(1234)
    walk = walkthrough(spec)
    after = ["look", "inventory", "go north", "take unicorn", ""]
    rand = lambda k: [None] * k  # noqa: E731  (None: a random command, chosen when it is played)
    episodes = [
        # the walkthrough with probes between moves, then commands after the game has ended
        [c for cmd in walk for c in (cmd, "look")] + after,
        probe_commands(spec) + rand(120) + walk + after,
        rand(150),
        walk + after,
    ]
    # eat each forbidden food (a fail event): it lies in the starting room in the games made here
    infos = dict((eid, info) for eid, info in spec["infos"])
    for q in spec["quests"]:
        for fe in q["fail_events"]:
            for p in fe["condition"]["preconditions"]:
                food = infos[p["arguments"][0]["name"]]["name"]
                episodes.append(["eat " + food, "take " + food, "inventory", "eat " + food] + after)
    sim = FastTextWorldSimulator(spec_path)
    observations, states = Table(), Table()
    privileged = lambda sim: states(sim.get_privileged_state())  # noqa: E731
    result = []
    for commands in episodes:
        obs, info = sim.reset()
        ep = {"reset": {"obs": observations(obs), "info": info, "privileged": privileged(sim)}, "steps": []}
        for i, cmd in enumerate(commands):
            if cmd is None:
                cmd = random_command(spec, sim, rng)
            obs, reward, done, info = sim.step(cmd)
            step = {"cmd": cmd, "obs": observations(obs), "reward": reward, "done": done, "info": info}
            if i % 10 == 0 or i == len(commands) - 1:
                step["privileged"] = privileged(sim)
            ep["steps"].append(step)
        result.append(ep)
    return {"observations": observations.values, "privileged": states.values, "episodes": result}


def main(out_dir, check_only=False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not check_only:
        for name, seed, rooms, objects, length, breadth, fail in GAMES:
            spec = make_spec(seed, rooms, objects, length, breadth)
            if fail:
                add_fail_event(spec)
            (out / f"{name}.json").write_text(json.dumps(spec, separators=(",", ":")))
    for name, *_ in GAMES:
        spec_path = out / f"{name}.json"
        transcript = {"game": spec_path.name, **run(str(spec_path))}
        text = json.dumps(transcript, separators=(",", ":"))
        target = out / f"{name}.transcript.json"
        if check_only:
            if target.read_text() != text:
                sys.exit(f"{target} differs under PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}")
        else:
            target.write_text(text)
            steps = sum(len(e["steps"]) for e in transcript["episodes"])
            print(f"{name}: {steps} steps, {len(text)} bytes")
    if not check_only:
        # the transcripts must not depend on the string hash seed
        for seed in ("1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            subprocess.run([sys.executable, __file__, out_dir, "--check"], env=env, check=True)
        print("transcripts are identical under other hash seeds")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test/fixtures/rl", "--check" in sys.argv)
