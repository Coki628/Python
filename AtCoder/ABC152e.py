# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/abc152/submissions/9609584
・自力ならず。。
・結局愚直LCM解は想定解じゃなかったんだけど、MOD取りをLCMの途中でやらないで、
　LCM終わった最後だけにすれば、後は除算とか適切にやるだけで普通に通った。。
・つまりLCM(GCD)の計算はMOD取った後やるとバグるけど、
　LCMが全部終わった後でMOD取る分には、後は普通の四則計算だからOK、ってことか。。
・制約N<=1万で、pythonAC1.0秒、pypyAC0.9秒。
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

def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def lcm(x, y): return (x * y) // gcd(x, y)

N = INT()
A = LIST()

l = 1
for a in A:
    l = lcm(l, a)
l %= MOD

B = [0] * N
for i, a in enumerate(A):
    B[i] = fermat(l, a, MOD)

ans = 0
for b in B:
    ans += b
    ans %= MOD
print(ans)
