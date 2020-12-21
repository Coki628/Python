# -*- coding: utf-8 -*-

"""
・自力AC
・一見2乗ループしたくなるけど、よく考察して法則が分かると一発でいけるやつ。
・強い人はみんな一瞬で見抜くんだよなぁこういうの。
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

N, K = MAP()
S = input()

L = S[:K-1]
R = S[K-1:]
# 操作回数の偶奇によって左はリバースさせる
if (N-K+1) % 2 == 1:
    L = L[::-1]
ans = R + L
print(ans)
