# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc130/editorial.pdf
　　　https://atcoder.jp/contests/abc130/submissions/5979643
・LCSっぽいものを、最長じゃなくて、数え上げる。
・dpの中身を2次元累積和にして、動的に更新していく。
・累積和更新と時に+-する添字位置は、普通の区間取得の時とは違う。
　どうなってるかは自分のメモ参照
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

N,M=MAP()
S=[0]+LIST()
T=[0]+LIST()

dp=[[0]*(M+1) for i in range(N+1)]
for i in range(1, N+1):
    for j in range(1, M+1):
        # dp配列を2次元累積和にして、動的に更新していく
        # (+-する添字位置は、普通の区間取得の時とは違う。どうなってるかは自分のメモ参照)
        dp[i][j]=(dp[i-1][j]+dp[i][j-1]-dp[i-1][j-1])%MOD
        # 今回増やす数値が一致するなら
        if S[i]==T[j]:
            # そこより手前の全部合計+1
            dp[i][j]+=dp[i-1][j-1]+1
            dp[i][j]%=MOD
# 全部の合計+最初の空配列ペアの分
print((dp[N][M]+1)%MOD)
