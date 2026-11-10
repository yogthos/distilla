"""Play compiled TextWorld games (.z8, Inform7) for distilla's RL environment.

distilla runs the agent, the memory and the training in jolt; TextWorld's
Inform7 games need TextWorld's own interpreter, which this script drives. One
JSON request per line on stdin, one JSON reply per line on stdout:

    {"op": "reset", "game": "path/to/game.z8"}  -> {"feedback": ...}
    {"op": "step", "command": "go north"}        -> {"feedback", "reward", "done", "score", "won"}
    {"op": "close"}                              -> exits

Needs `pip install textworld==1.7.0`.
"""

import json
import sys

import textworld


def feedback(state):
    text = state.feedback
    return text.decode("utf-8", errors="replace") if isinstance(text, bytes) else str(text)


def main() -> None:
    env = None
    for line in sys.stdin:
        request = json.loads(line)
        op = request.get("op")
        if op == "reset":
            if env is not None:
                env.close()
            env = textworld.start(request["game"])
            reply = {"feedback": feedback(env.reset())}
        elif op == "step":
            state, reward, done = env.step(request["command"])
            reply = {
                "feedback": feedback(state),
                "reward": float(reward),
                "done": bool(done),
                "score": int(getattr(state, "score", 0) or 0),
                "won": bool(getattr(state, "won", False)),
            }
        elif op == "close":
            break
        else:
            reply = {"error": f"unknown op {op!r}"}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
