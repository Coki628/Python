# -*- coding: utf-8 -*-

"""
・2次元目の添字をdictで管理するDP
・TLE、部分点獲得。
"""

import sys
from collections import defaultdict

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

N,D=MAP()
X,Y=MAP()

dp=[defaultdict(int) for i in range(N+1)]
dp[0][(0, 0)]=1
for i in range(N):
    for k, v in dp[i].items():
        x,y=k
        dp[i+1][(x+D, y)]+=v
        dp[i+1][(x, y+D)]+=v
        dp[i+1][(x-D, y)]+=v
        dp[i+1][(x, y-D)]+=v
total=0
scs=0
for k, v in dp[N].items():
    total+=v
    if k==(X, Y):
        scs=v
print(scs/total)
