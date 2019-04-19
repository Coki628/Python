# -*- coding: utf-8 -*-

"""
参考：http://ddcc2016-qual.contest.atcoder.jp/data/other/ddcc2016-qual/editorial.pdf
　　　https://atcoder.jp/contests/ddcc2016-qual/submissions/3617098
・円の方程式使わなくても、三平方でいけたんだねー。
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
    # 縦の長さ(円の中心から、i番目の切り込みのy座標位置までの距離)
    y=2*r/N*i
    b=abs(r-y)
    # 横の長さ(斜辺は常に半径rでOK)
    a=sqrt(r**2-b**2)
    # 各位置のカット長(aは半径までなので*2する)
    lns.append(a*2)
lns+=[0]*N

ans=0
for i in range(M, N*3):
    ans+=max(lns[i-M], lns[i])
print(ans)
