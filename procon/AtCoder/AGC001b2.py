# -*- coding: utf-8 -*-

"""
参考：http://agc001.contest.atcoder.jp/data/agc/001/editorial.pdf
・500点自力ならず。
・奇数の(N-1)*3が違うってなった時、ちょっと素数かどうか、
　は考えて素数じゃない9を見たんだけど、手元で9,2までしか試さなかった。
　何で9,3やらなかったんだろう。まあ、それで違うって分かったところで、
　どうすれば正しくできるかでGCDに辿り着けた可能性は低いけどね。。
・でも、少なくともN=8と9までは手元で全部試すべきだったな。
"""

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd

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

N, X = MAP()

print((N-gcd(N, X))*3)
