# -*- coding: utf-8 -*-

"""
・蟻本演習3-1-1、自力AC！
・二分探索
・循環する系は2周作る。ってやったけど、今回はD位置の本店だけ増やせばよかった。。
"""

import sys
from bisect import bisect_left

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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

D = INT()
N = INT()
M = INT()

A = [0]
for i in range(N-1):
    A.append(INT())
A.sort()
# 前後の番兵 + 2周分作る
A = [-INF] + A + [a+D for a in A] + [INF]

ans = 0
for _ in range(M):
    b = INT()
    idx = bisect_left(A, b)
    dist = min(abs(b-A[idx-1]), abs(b-A[idx]))
    ans += dist
print(ans)
