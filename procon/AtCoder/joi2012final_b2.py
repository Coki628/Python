# -*- coding: utf-8 -*-

"""
・自力AC！
・LCS系の2方向indexのDP
・2方向DPいつもあんまり自力で解けない感じだったので、自力AC嬉しい。
・一度編成を開始したら途中スキップができない条件なので、使わないの遷移をしない。
・開始位置、終了位置はIでないといけないので、初期化と最後の最大値取得はその条件に合わせて行う。
・TLEしてたのを、3次元配列を2次元2つにして無事AC。計算量400万でもpypyAC0.6秒。
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

N, M = MAP()
S = input()
T = input()

# dp0[i][j] := Sをi文字目、Tをj文字目まで見て、最後に使ったのがIの時の最大の長さ
dp0 = list2d(N+1, M+1, -INF)
# dp1[i][j] := Sをi文字目、Tをj文字目まで見て、最後に使ったのがOの時の最大の長さ
dp1 = list2d(N+1, M+1, 0)
for i in range(N+1):
    for j in range(M+1):
        if i < N:
            # Sから1文字使う
            if S[i] == 'I':
                dp0[i+1][j] = max(dp0[i+1][j], dp1[i][j] + 1)
            else:
                dp1[i+1][j] = max(dp1[i+1][j], dp0[i][j] + 1)
        if j < M:
            # Tから1文字使う
            if T[j] == 'I':
                dp0[i][j+1] = max(dp0[i][j+1], dp1[i][j] + 1)
            else:
                dp1[i][j+1] = max(dp1[i][j+1], dp0[i][j] + 1)
ans = 0
for i in range(N+1):
    # 終了地点がI(直前がI)のものから最大を取る
    ans = max(ans, max(dp0[i]))
print(ans)
