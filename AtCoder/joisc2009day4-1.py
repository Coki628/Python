# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・基本、書いてある通りに作るだけ。設定いっぱいあってめんどくさいけど。
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
MOD = 10 ** 5

N, M, T, X, Y = MAP()
P = LIST(M)

def calc(t, n, m):
    return max(P[m] - (t-op[m][n]) - 120*wa[m][n], X)

op = list2d(M, N, 0)
wa = list2d(M, N, 0)
ans = [0] * N
for i in range(Y):
    t, n, m, s = input().split()
    t, n, m = int(t), int(n) - 1, int(m) - 1
    if s == 'open':
        op[m][n] = t
    elif s == 'correct':
        ans[n] += calc(t, n, m)
    else:
        wa[m][n] += 1
[print(a) for a in ans]
