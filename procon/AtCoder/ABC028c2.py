# -*- coding: utf-8 -*-

from itertools import combinations

A, B, C, D, E = map(int, input().split())

# これだけでよかったなんて…
print(max(A+D+E, B+C+E))