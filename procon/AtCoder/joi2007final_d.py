# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-13
・グリッドDP、って紹介されてたけど内容的にはグリッドに沿ってのDPはしてない。
・制約が小さくて、添字が多くて動きがややこしい系。普通の1方向DPに色々枝葉が付いた感じ。
・3次元の内側で2種類の遷移ループ。
・なかなかWAが取れなくて、色々デバッグしたり、結局人の提出見比べたりした。
・原因は主に最初に1段飛ばしする場合と最後に1段飛ばしする場合を正しく遷移させてなかったこと。
・計算量はN*M*K^2=150*75*100=112万くらいで、pythonTLE(制約1秒)、pypyは最初MLE、
　INFをfloat('inf')から10**18にして、0.3秒AC。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

# stones[(i, j)] := (列番号, 滑りやすさ)
stones = list2d(N+1, 11, None)
stones[0][0] = (0, 0)
for i in range(N):
    li = LIST()
    m, li = li[0], li[1:]
    for j in range(0, m*2, 2):
        stones[i+1][j//2] = (li[j], li[j+1])

# dp[i][j][k] := i行目で1つ飛ばしをj回していて、k番目の石にいる状態での危険度の最小値
dp = list3d(N+1, M+1, 10, INF)
for i in range(10):
    dp[0][0][i] = 0

for i in range(N):
    for j in range(M+1):
        k = 0
        while stones[i][k]:
            x, d = stones[i][k]
            # 普通のジャンプの遷移
            l = 0
            while stones[i+1][l]:
                x2, d2 = stones[i+1][l]
                # 最初のジャンプは無条件で危険度0
                if i == 0:
                    dp[i+1][j][l] = min(dp[i+1][j][l], dp[i][j][k])
                else:
                    dp[i+1][j][l] = min(dp[i+1][j][l], dp[i][j][k] + (d+d2)*abs(x-x2))
                l += 1
            # 1つ飛ばしジャンプの遷移
            if i+2 <= N and j+1 <= M:
                l = 0
                while stones[i+2][l]:
                    x2, d2 = stones[i+2][l]
                    # 最初のジャンプは無条件で危険度0
                    if i == 0:
                        dp[i+2][j+1][l] = min(dp[i+2][j+1][l], dp[i][j][k])
                    else:
                        dp[i+2][j+1][l] = min(dp[i+2][j+1][l], dp[i][j][k] + (d+d2)*abs(x-x2))
                    l += 1
            k += 1

ans = INF
for j in range(M+1):
    for k in range(10):
        ans = min(ans, dp[N][j][k])
# 最後に1つ飛ばしジャンプでゴールする場合
for j in range(M):
    for k in range(10):
        ans = min(ans, dp[N-1][j][k])
print(ans)
