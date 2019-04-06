# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc123/editorial.pdf
　　　https://www.youtube.com/watch?v=KwHd_LKcnrY
・公式解3：優先度付キュー
・これも速い。0.03秒くらい。
"""

import sys
from heapq import heappush, heappop

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

X,Y,Z,K=MAP()
A=sorted(LIST(), reverse=True)
B=sorted(LIST(), reverse=True)
C=sorted(LIST(), reverse=True)


que=[]
heappush(que, (-(A[0]+B[0]+C[0]), 0, 0, 0))
S={(0,0,0)}
for i in range(K):
    ans=heappop(que)
    print(-ans[0])
    a,b,c=ans[1:]
    if a+1<X and (a+1,b,c) not in S:
        heappush(que, (-(A[a+1]+B[b]+C[c]), a+1, b, c))
        S.add((a+1,b,c))
    if b+1<Y and (a,b+1,c) not in S:
        heappush(que, (-(A[a]+B[b+1]+C[c]), a, b+1, c))
        S.add((a,b+1,c))
    if c+1<Z and (a,b,c+1) not in S:
        heappush(que, (-(A[a]+B[b]+C[c+1]), a, b, c+1))
        S.add((a,b,c+1))
