# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc130/editorial.pdf
・LCSっぽいものを、最長じゃなくて、数え上げる。
・これは和を愚直で取ってるのでTLE想定。
・ここから内側のループを2次元累積和で高速化する。
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
S=LIST()
T=LIST()

dp=[[0]*M for i in range(N)]
for i in range(N):
    for j in range(M):
        # 今回増やす数値が一致するなら
        if S[i]==T[j]:
            sm=0
            for k in range(i):
                for l in range(j):
                    sm+=dp[k][l]
            # そこより手前の全部合計+1
            dp[i][j]+=sm+1
ans=0
for i in range(N):
    for j in range(M):
        ans+=dp[i][j]
# 全部の合計+最初の空配列ペアの分
print(ans+1)
