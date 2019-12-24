# -*- coding: utf-8 -*-

"""
・自力ならず。
・回文、GCD系
・何かGCD使いそうだなーって思ってa-mnからGCD取るところまではやってた。
・とはいえそれも雰囲気でやっただけだし、その先はもう出なかったと思うから、
　ちゃんと論理立てて導けるようにしたいよね。
"""

import sys
from fractions import gcd

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
A = LIST(N)

mn = min(A)
g = 0
for a in A:
    g = gcd(g, a-mn)

# 先端mnの選び方と、真ん中はgcd周期での選び方(長さgを回文にするので、/2切り上げ)
print(pow(2, mn + ceil(g, 2), MOD))
