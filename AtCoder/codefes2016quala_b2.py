# -*- coding: utf-8 -*-

"""
参考：http://code-festival-2016-quala.contest.atcoder.jp/data/other/code-festival-2016-quala/editorial.pdf
・絶対シンプルに解く方法あるよなー、と思って、やっぱりあったよの公式解。
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

cnt=0
for i in range(N):
    if A[A[i]-1]==i+1:
        cnt+=1
print(cnt//2)
