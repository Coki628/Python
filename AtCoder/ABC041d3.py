# -*- coding: utf-8 -*-

"""
・蟻本演習3-4-1、自力AC！
・bitDP、トポロジカルソートの数え上げ
・pythonAC0.4秒、pypyAC0.2秒。
"""

import sys

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

N, M = MAP()
XY = []
nodes = [0] * N
for i in range(M):
    x, y = MAP()
    x -= 1; y -= 1
    XY.append((x, y))
    # 自分より後ろなのが確定している頂点の集合を持たせる
    nodes[x] |= 1 << y

dp = [0] * (1<<N)
dp[0] = 1
for bit in range(1<<N):
    for i in range(N):
        if bit >> i & 1:
            continue
        # 今の集合bitに、自分より後ろであるべき頂点が1つも含まれていなければOK
        if not (bit & nodes[i]):
            dp[bit|1<<i] += dp[bit]
print(dp[-1])
