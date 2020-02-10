# -*- coding: utf-8 -*-

"""
・500点自力AC！典型を手早く通せて良かった。
・桁DP
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

S = input()
N = len(S)
K = INT()

# dp0[i][j] := i桁目まで見て、0以外の数字がj個ある値の通り数(smaller確定前)
dp0 = list2d(N+1, 107, 0)
# dp1[i][j] := i桁目まで見て、0以外の数字がj個ある値の通り数(smaller確定後)
dp1 = list2d(N+1, 107, 0)
dp0[0][0] = 1
for i in range(N):
    num = int(S[i])
    for j in range(107):
        for k in range(10):
            nxt = j+1 if k != 0 else j
            if nxt == 107:
                continue
            if k < num:
                dp1[i+1][nxt] += dp0[i][j]
                dp1[i+1][nxt] += dp1[i][j]
            elif k == num:
                dp0[i+1][nxt] += dp0[i][j]
                dp1[i+1][nxt] += dp1[i][j]
            else:
                dp1[i+1][nxt] += dp1[i][j]
ans = dp0[N][K] + dp1[N][K]
print(ans)
