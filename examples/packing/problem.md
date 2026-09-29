# Spread ten points out

Place **10 points in the unit square** so that the two closest of them are as far apart as
possible.

Write `solution.py`. Running it, with `python solution.py`, must print one line to stdout: a
JSON list of exactly 10 points, `[[x, y], ...]`, with every coordinate between 0 and 1
inclusive. It may use the Python standard library and nothing else, and it has 30 seconds.

**The score is the smallest distance between any two of the points. Larger is better.**

An answer that prints something else, has the wrong number of points, or puts a point
outside the square is not scored at all.

For a sense of scale: ten points on a 4-by-3 grid score `1/3`; the best arrangement known
scores about `0.4213`. There is room between the two, and a model that only tunes the grid
will not find it.

The scorer is `score.py` beside this file. It is deterministic, so the same program always
scores the same, and it takes well under a second.
