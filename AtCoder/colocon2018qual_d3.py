# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/colopl2018-qual/editorial.pdf
　　　http://tutuz.hateblo.jp/entry/2018/10/27/163217
　　　http://seica-at.hatenablog.com/entry/2017/12/10/173410
・配るDP、minmax系
・ひとつ右下とその下全体のうち、最適な2か所にmaxを送る。
・pypyで時間もメモリもギリギリのAC
・添え字の+1とか-1の調整がすげーめんどくさかった。
・配列Tの先頭に番兵入れたらだいぶ見やすくなった。
・計算量O(N^2*2)=5000^2*2=5000万+jの位置を探すwhileの分
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
# 1-indexedに揃える(あとで見やすい)
T=[0]+[INT() for i in range(N)]

# dp[i][k] := i個までの時刻のうち、k回起動しているときの知力の最大値
dp=list2d(N+1, N+1, 0)
for i in range(1, N+1):
    # 初期化：開始時スタミナ満タンなので、1回起動なら確実にX使える
    dp[i][1]=X

for i in range(1, N):
    j=i+1
    # T[j]<=T[i]+x なる最大の j
    while j+1<N and T[j+1]<=T[i]+X: j+=1
    for k in range(1, i+1):
        # 配る前に左から貰ってきておく
        dp[i][k]=max(dp[i][k], dp[i][k-1])
        # 最適な2か所に配る
        dp[j][k+1]=max(dp[j][k+1], dp[i][k]+min(X, T[j]-T[i]))
        if j<N:
            dp[j+1][k+1]=max(dp[j+1][k+1], dp[i][k]+min(X, T[j+1]-T[i]))

# 飛ばして遷移させていてN行目が最大とは限らないのでmaxを取る
for k in range(1, N+1):
    mx=0
    for i in range(1, N+1):
        mx=max(mx, dp[i][k])
    print(mx)
