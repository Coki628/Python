# -*- coding: utf-8 -*-

"""
参考：https://www.youtube.com/watch?v=8uowVvQ_-Mo
・ACしたけど、2019が素数じゃなかったから673*3使って0になった訳で、
　本当は2019*2回した方が良かったね…。
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

L,R=MAP()

if R-L+1>2019:
    print(0)
else:
    ans=INF
    for i in range(L, R+1):
        for j in range(i+1, R+1):
            ans=min(ans, (i%2019)*(j%2019)%2019)
    print(ans)
