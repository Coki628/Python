# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc129/editorial.pdf
　　　https://www.youtube.com/watch?v=L8grWxBlIZ4
・XOR系
・桁DP
・2つの状態をうまく使って4通りの場合分けで組の個数を遷移させる。
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

L=input()
N=len(L)

# dp[i][j] := i桁目まで考えた時、j=0はi桁目まで一致(L=A+B)、j=1は既にA+BがL未満(L>A+B)になった状態
dp=list2d(N+1, 2, 0)
# 0桁目(始まる前)は同じとみなしてj=0に1通り
dp[0][0]=1

for i in range(N):
    # i桁目のa+bを0にする場合
    # Lのi桁目が0なら
    if L[i]=='0':
        # 何も変化しない
        dp[i+1][0]+=dp[i][0]
        dp[i+1][1]+=dp[i][1]
    # Lのi桁目が1なら
    else:
        # これまでL=A+BだったものもL>A+Bになるので、j=1もj=0もj=1に遷移させる
        dp[i+1][1]+=dp[i][0]+dp[i][1]
    dp[i+1][0]%=MOD
    dp[i+1][1]%=MOD

    # i桁目のa+bを1にする場合
    # Lのi桁目が0なら
    if L[i]=='0':
        # 既にL>A+Bだったものだけが遷移できる(aを1とするかbを1とするかで*2)
        dp[i+1][1]+=dp[i][1]*2
    # Lのi桁目が1なら
    else:
        # 状態は変化しない(ただしaを1とするかbを1とするかで*2)
        dp[i+1][0]+=dp[i][0]*2
        dp[i+1][1]+=dp[i][1]*2
    dp[i+1][0]%=MOD
    dp[i+1][1]%=MOD
# 最終的にL=A+Bだった分とL>A+Bだった分を合計する
print((dp[N][0]+dp[N][1])%MOD)
