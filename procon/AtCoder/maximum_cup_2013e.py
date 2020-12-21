# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/maximum-cup-2013/submissions/10029910
・自力ならず。予め除外しておくやつを考慮してなかった。
・確率、期待値
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, t = MAP()

# 10枚に1回必ずNが出る分を予め回数から引いておく
t -= t // 10
total = 0
rare = 0
for i in range(N):
    name, rank, rate = input().split()
    rate = int(rate)
    if name.count('Alicia') and rank != 'N':
        rare += rate
    total += rate
ans = rare / total * t
print(ans)
