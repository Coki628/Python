# -*- coding: utf-8 -*-

"""
・円の方程式
・桁数が大きい時、小数点以下が誤差出てる感じだったけど普通に通った。
"""

import sys
from math import sqrt

def input(): return sys.stdin.readline().strip()
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

r,N,M=MAP()

lns=[0]*N
for i in range(N):
    y=2*r/N*i
    # 円の方程式から、xについて解く
    x1=-sqrt(r**2-(y-r)**2)+r
    x2=sqrt(r**2-(y-r)**2)+r
    # 各位置のカット長
    lns.append(abs(x1-x2))
lns+=[0]*N

ans=0
for i in range(M, N*3):
    ans+=max(lns[i-M], lns[i])
print(ans)
