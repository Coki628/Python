# -*- coding: utf-8 -*-

"""
・部分点想定
・優先度付きキュー
・増築する度に金額を上げてキューに入れ直す、を繰り返す。
"""

import sys
from heapq import heapify, heappop, heappush

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

K = INT()
N = INT()
AD = []
for i in range(N):
    a, d = MAP()
    AD.append((a, d))

heapify(AD)
k = 0
ans = 0
while k < K:
    a, d = heappop(AD)
    ans += a
    a += d
    heappush(AD, (a, d))
    k += 1
print(ans)
