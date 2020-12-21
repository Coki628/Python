# -*- coding: utf-8 -*-

"""
・実験用。やったけど何も見えず。。
"""

import sys
from itertools import combinations
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

H, W, K = MAP()

N = H * W
ans = 0
C = Counter()
for comb in combinations(range(N), K):
    HW = []
    for hw in comb:
        h, w = hw//W, hw%W
        HW.append((h, w))
    sm = 0
    for i in range(K):
        for j in range(i+1, K):
            h1, w1 = HW[i]
            h2, w2 = HW[j]
            sm += abs(h1-h2) + abs(w1-w2)
    C[sm] += 1
    ans += sm
print(*C.items(), end=', ')
print(ans)
