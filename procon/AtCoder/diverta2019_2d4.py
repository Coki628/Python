# -*- coding: utf-8 -*-

"""
参考：https://misteer.hatenablog.com/entry/diverta2019-2-D
・個数制限なしナップザック
・金銀銅を使ったドングリの換金は、品物が3つのナップザックDPとみなせる。
・A→Bでの換金とB→Aでの換金の2回分、DPする。
・計算量はN^2の2500万と定数3で7500万、このままだとTLE。。
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
A = LIST()
B = LIST()

def solve(N):
    # dp[i][j] := 金銀銅のうちi個目まで見て、ドングリをj個使った時に得られるドングリの最大値
    dp = list2d(4, N+1, -INF)
    # 初期化、換金をしないことで、i個は必ず得られる
    for i in range(N+1):
        dp[0][i] = i
    for i in range(3):
        a, b = A[i], B[i]
        for j in range(N+1):
            dp[i+1][j] = max(dp[i+1][j], dp[i][j])
            if j+a <= N:
                # 個数制限なしなので、iの同じ段にも遷移させる
                dp[i][j+a] = max(dp[i][j+a], dp[i][j] + b)
                dp[i+1][j+a] = max(dp[i+1][j+a], dp[i][j] + b)
    return dp[3][N]

# 前半
N2 = solve(N)
# 換金のコストと利益を入れ替え
A, B = B, A
# 後半
ans = solve(N2)
print(ans)
