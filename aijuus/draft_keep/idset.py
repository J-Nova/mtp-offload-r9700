#!/usr/bin/env python3
"""Single id-list parser shared by the draft-keep builders (build_vocab.py, merge.py).

Accepts a JSON list of ints (the collector's rank*.json) or any whitespace-separated ints (a seed
list). Kept in one place so the two derived keep files cannot drift in how they read the collector
output.
"""
import json


def read_ids(path):
    s = open(path).read()
    try:
        d = json.loads(s)
        if isinstance(d, list):
            return [int(x) for x in d if not isinstance(x, bool)]
    except Exception:
        pass
    out = []
    for tok in s.split():
        try:
            out.append(int(tok))
        except ValueError:
            pass
    return out
