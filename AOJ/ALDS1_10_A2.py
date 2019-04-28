# -*- coding: utf-8 -*-

"""
・DP基礎
・フィボナッチ数列
・メモ化再帰
"""

import sys
sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))

N=INT()

memo=[0]*50
memo[0]=memo[1]=1
def rec(n):
    if memo[n]:
        return memo[n]
    memo[n]=rec(n-1)+rec(n-2)
    return memo[n]

print(rec(N))
