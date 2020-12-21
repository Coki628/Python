# -*- coding: utf-8 -*-

"""
・defaultdictが遅かったので、indexずらして普通の配列で持ってみた。速くなった。
・pythonAC1.2秒、pypyAC0.2秒。
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

dp0 = list2d(N+1, 4, -INF)
dp1 = list2d(N+1, 4, -INF)
dp0[0][2] = 0
for i, a in enumerate(A):
    for j in range(4):
        if j != 0:
            dp0[i+1][j-1] = max(dp0[i+1][j-1], dp0[i][j])
            dp0[i+1][j-1] = max(dp0[i+1][j-1], dp1[i][j])
        if j != 3:
            dp1[i+1][j+1] = max(dp1[i+1][j+1], dp0[i][j] + a)
if N % 2 == 0:
    ans = max(dp0[N][2], dp1[N][2])
else:
    ans = max(dp0[N][1], dp1[N][1])
print(ans)
