# -*- coding: utf-8 -*-

"""
・激戦の末、なんとか自力AC！
・再帰探索(メモ化再帰)、GCD,LCM系
・再帰で潜って、一番下から順番に決めていく。
・最上段がどこか分からないので、終わった範囲はメモしておいて全部からやる。
・ここまでは割とすぐ作れたんだけど、その段の重さを決める計算パートがすげー時間かかった。
・色々例作って頑張って考えて、最終的にLCM出して比で割る、に行き着いたけど、
　これ数学強いと一瞬で出るんだろーな、うらやま。。
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
A = []
for _ in range(N):
    p, q, a, b = MAP()
    g = gcd(p, q)
    p //= g; q //= g
    a -= 1; b -= 1
    A.append((p, q, a, b))

def calc(p, q, w1, w2):
    l = lcm(p*w1, q*w2)
    return l // p + l // q

memo = [-1] * N
def rec(i):
    if i == -1:
        return 1
    if memo[i] != -1:
        return memo[i]
    p, q, a, b = A[i]
    w1 = rec(a)
    w2 = rec(b)
    res = calc(p, q, w1, w2)
    memo[i] = res
    return res

for i in range(N):
    rec(i)
print(max(memo))
