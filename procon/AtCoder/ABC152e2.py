# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/abc152/submissions/9630459
・LCMで一切MOD取らなくても通る解。
・定数倍高速化として、
　・MOD取りを都度じゃなくて最後だけにする。
　・配列B作れないで直接答えに足す。
　をやったらpythonAC1.9秒、pypyTLE。
・MOD取り最後だけのが速いのはかなり意外。四則計算の中でMOD遅いのは聞いてたけど、
　さすがに際限なく増える桁の加算がそれより速いのは分からないよなー。。
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

def lcm(x, y): return (x * y) // gcd(x, y)

N = INT()
A = LIST()

l = 1
for a in A:
    l = lcm(l, a)

ans = 0
for i, a in enumerate(A):
    ans += l // a
print(ans%MOD)
