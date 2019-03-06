# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/colopl2018-qual/editorial.pdf
　　　http://tutuz.hateblo.jp/entry/2018/10/27/163217
・ナイーブな遷移をした場合
・これはTLE前提だけど、練習のために書いた。
・配るDP、minmax系
・ひとつ右下とその下全体にmaxを送る。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N,X=MAP()
# 1-indexedに揃える
T=[0]+[INT() for i in range(N)]

# dp[i][k] := i個までの時刻のうち、k回起動しているときの知力の最大値
dp=list2d(N+1, N+1, 0)
for i in range(1, N+1):
    # 初期化：開始時スタミナ満タンなので、1回起動なら確実にX使える
    dp[i][1]=X

for i in range(1, N):
    for k in range(1, i+1):
        for j in range(i+1, N+1):
            dp[j][k+1]=max(dp[j][k+1], dp[i][k]+min(X, T[j]-T[i]))

[print(dp[N][i]) for i in range(1, N+1)]
