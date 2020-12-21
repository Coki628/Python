# -*- coding: utf-8 -*-

"""
・自力AC
・全探索
・計算量100C2*100=約50万で制約1秒だとどうかなーと思ったけど、
　意外とpythonでも余裕のAC0.2秒。
"""

import sys
from itertools import combinations

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, M = MAP()
table = [[]] * N
for i in range(N):
    table[i] = LIST()

ans = 0
for comb in combinations(range(M), 2):
    a, b = comb
    sm = 0
    for i in range(N):
        sm += max(table[i][a], table[i][b])
    ans = max(ans, sm)
print(ans)
