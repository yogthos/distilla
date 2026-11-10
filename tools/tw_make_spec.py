"""Write a TextWorld game spec (.json, for the in-process simulator) without compiling it.

    python tools/tw_make_spec.py OUT.json SEED ROOMS OBJECTS QUEST_LENGTH QUEST_BREADTH
"""

import sys

import textworld
from textworld.generator import make_game

path, seed, rooms, objects, length, breadth = sys.argv[1], *map(int, sys.argv[2:7])
options = textworld.GameOptions()
options.seeds = seed
options.nb_rooms = rooms
options.nb_objects = objects
options.quest_length = length
options.quest_breadth = breadth
make_game(options).save(path)
