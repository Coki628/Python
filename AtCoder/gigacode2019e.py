# -*- coding: utf-8 -*-

"""
・自力AC
・添字にN、遷移にNでN^2のDP。制約はN<=2000くらいなのでこれでOK。
・pythonAC1.0秒、pypyAC0.3秒。
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

N, L = MAP()
v, d = MAP()
XVD = [(0, v, d)]
X = [0]
for _ in range(N):
    x, v, d = MAP()
    XVD.append((x, v, d))
    X.append(x)
X.append(L)
XVD.sort()
X.sort()

# dp[i] := i番目の車の位置まで辿り着くための最短時間
dp = [INF] * (N+2)
dp[0] = 0
for i, (x1, v, d) in enumerate(XVD):
    for j, x2 in enumerate(X[i+1:], i+1):
        dist = x2 - x1
        if dist > d:
            break
        time = dist / v
        dp[j] = min(dp[j], dp[i] + time)
ans = dp[N+1]
if ans == INF:
    print('impossible')
else:
    print('{:.10f}'.format(ans))
