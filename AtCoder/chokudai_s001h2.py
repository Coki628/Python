# -*- coding: utf-8 -*-

"""
参考：https://ikatakos.com/pot/programming_algorithm/dynamic_programming/longest_common_subsequence
・LIS(最長増加部分列)
・ソート済のAをBとすれば、AとBのLCSに帰着できる。(今回は制約的にダメぽいけど)
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

N=INT()
A=LIST()
B=sorted(A)

dp = [[0] * (N+1) for i in range(N+1)]

for i in range(N+1):
    for j in range(N+1):
        # 下に送る
        if i < N:
            dp[i+1][j] = max(dp[i+1][j], dp[i][j])
        # 右に送る
        if j < N:
            dp[i][j+1] = max(dp[i][j+1], dp[i][j])
        # 今回増やす文字が一致するなら、LCSが1文字増える
        if i < N and j < N and A[i] == B[j]:
            # 右下に送る
            dp[i+1][j+1] = max(dp[i+1][j+1], dp[i][j]+1)

print(dp[N][N])
