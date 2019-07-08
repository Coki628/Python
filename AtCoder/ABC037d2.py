# -*- coding: utf-8 -*-

"""
参考：http://abc037.contest.atcoder.jp/data/abc/037/editorial.pdf
・グリッドでメモ化再帰
・これは自力で解きたかったな。。
・なんか最初pypyでも1ケースだけTLEで、4方向探しに行く順番変えたら通った。謎。。
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

H,W=MAP()
# 四方に一回り大きいグリッドを作る
grid = list2d(H+2, W+2, INF)
directions = ((1,0),(-1,0),(0,1),(0,-1))
for i in range(1, H+1):
    row = LIST()
    for j in range(1, W+1):
        grid[i][j] = row[j-1]

memo=list2d(H+2, W+2, 0)
def rec(h1, w1):
    if memo[h1][w1]:
        return memo[h1][w1]
    res=1
    for h, w in directions:
        h2=h1+h
        w2=w1+w
        if grid[h2][w2]!=INF and grid[h2][w2]>grid[h1][w1]:
            res+=rec(h2, w2)
            res%=MOD
    memo[h1][w1]=res
    return res

ans=0
for i in range(1, H+1):
    for j in range(1, W+1):
        ans+=rec(i, j)
        ans%=MOD
print(ans)
