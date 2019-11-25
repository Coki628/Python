# -*- coding: utf-8 -*-

"""
・ぱっと浮かばなかったから飛ばした。
・終了後落ち着いて取り組んで自力AC。ちゃんとやればよかった。。
・1回操作すると、真ん中に1近づくけど真ん中が0.5ずれる(離れる)と考えられるので、
　1回の操作につき0.5ずつ近づくとすると、真ん中への距離(前後近い方)*2でOK。
"""

import sys
from bisect import bisect_left
from itertools import accumulate

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

N = INT()
A = [0] + LIST()

acc = list(accumulate(A))
mid = sum(A) / 2
l = bisect_left(acc, mid) - 1
r = bisect_left(acc, mid)
l = acc[l]
r = acc[r]
print(int(min(mid-l, r-mid)*2))
