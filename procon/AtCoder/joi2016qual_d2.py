# -*- coding: utf-8 -*-

"""
・並びを全列挙のTLE想定。
・内ループN=10万の一致比較をnumpyでまとめて処理。これで部分点2獲得。
"""

import sys
import numpy as np
from itertools import permutations
from collections import Counter

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
INF = float('inf')
MOD = 10 ** 9 + 7

N, M = MAP()
A = LIST(N)
A = np.array(A, dtype=np.int64)

C = Counter(A)
ans = INF
for perm in permutations(range(1, M+1)):
    good = []
    for a in perm:
        good += [a] * C[a]
    good = np.array(good, dtype=np.int64)
    # まとめて一致比較
    diff = (A!=good).sum()
    ans = min(ans, diff)
print(ans)
