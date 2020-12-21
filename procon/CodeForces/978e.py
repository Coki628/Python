# -*- coding: utf-8 -*-

"""
・結果から遡っていく。
・取りうる最大位置と最小位置を徐々に狭めていけば、開始時点まで戻った時に答えになってる。
・制約にN^2でDPなりやんなさいよって書いてあったから貪欲死ぬんじゃないかってビビりながら出したけど無事AC。
　なんで10万じゃなくてN<=1000にしたんだろう。
"""

import sys
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, W = MAP()
A = LIST()

lo = 0
hi = W
for a in A[::-1]:
    lo = max(lo-a, 0)
    hi = min(hi-a, W)
ans = max(hi - lo + 1, 0)
print(ans)
