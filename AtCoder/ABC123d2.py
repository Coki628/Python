# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc123/editorial.pdf
　　　https://www.youtube.com/watch?v=KwHd_LKcnrY
・400点自力ならず。
・公式解1：実は全探索2回に分けるだけでもいけたっていう…。
・300万のループがpythonTLEでpypyAC1.0秒くらい。
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

X,Y,Z,K=MAP()
A=LIST()
B=LIST()
C=LIST()

AB=[]
for i in range(X):
    for j in range(Y):
        AB.append(A[i]+B[j])
AB.sort(reverse=True)
# 上位3000位より下は考えなくていい。
# たとえCの1位と組んでも3001位以下になる。
AB=AB[:3000]

ABC=[]
for i in range(len(AB)):
    for j in range(Z):
        ABC.append(AB[i]+C[j])
ABC.sort(reverse=True)

for i in range(K):
    print(ABC[i])
