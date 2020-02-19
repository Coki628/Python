# -*- coding: utf-8 -*-

"""
・さくっと600点自力AC。
・なんか拍子抜け。最初何かやらないといけないと思ったけど、N<=50なら毎回max取ってindex探してを
　シミュレーションしても十分間に合うんじゃないかと思ったらそうだった。
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

N = INT()
A = LIST()

a = max(A)
cnt = 0
while a >= N:
    d, m = divmod(a, N)
    mxidx = A.index(a)
    for i in range(N):
        if i == mxidx:
            A[i] = m
        else:
            A[i] += d
    cnt += d
    a = max(A)
print(cnt)
