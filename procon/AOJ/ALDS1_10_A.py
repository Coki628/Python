# -*- coding: utf-8 -*-

"""
・DP基礎
・フィボナッチ数列
・普通のDP
"""

import sys
sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))

N=INT()

dp=[0]*50
dp[0]=dp[1]=1
for i in range(2, 50):
    dp[i]=dp[i-1]+dp[i-2]

print(dp[N])
