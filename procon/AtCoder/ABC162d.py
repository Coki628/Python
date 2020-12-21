# -*- coding: utf-8 -*-

"""
・ここまではさくっと自力AC
・累積和で前計算、2点決めうちで全探索
・計算量4000^2=1600万でpythonTLE,pypyAC0.2秒。
　やっぱり言語アップデートで速くなってる。
"""

import sys
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
S = input()
A = [0] * N
for i, s in enumerate(S):
    if s == 'G':
        A[i] = 1
    elif s == 'B':
        A[i] = 2

acc = list2d(3, N, 0)
for i, a in enumerate(A):
    acc[a][i] = 1
for i in range(3):
    acc[i] = list(accumulate(acc[i][::-1]))[::-1]

ans = 0
for i, a1 in enumerate(A):
    for j in range(i+1, N):
        a2 = A[j]
        if a1 == a2:
            continue
        a3 = 3 - a1 - a2
        cnt = acc[a3][j]
        dist = j - i
        if j + dist < N and A[j+dist] == a3:
            cnt -= 1
        ans += cnt
print(ans)
