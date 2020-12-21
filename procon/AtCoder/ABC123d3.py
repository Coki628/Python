# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc123/editorial.pdf
　　　https://www.youtube.com/watch?v=KwHd_LKcnrY
・公式解2：枝刈り探索
・1000の3重ループなのにめっちゃ速くなった。pythonで0.1秒くらい。
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
A=sorted(LIST(), reverse=True)
B=sorted(LIST(), reverse=True)
C=sorted(LIST(), reverse=True)

ABC=[]
for i in range(X):
    for j in range(Y):
        if (i+1)*(j+1)>K:
            break
        for k in range(Z):
            if (i+1)*(j+1)*(k+1)>K:
                break
            ABC.append(A[i]+B[j]+C[k])
ABC.sort(reverse=True)
for i in range(K):
    print(ABC[i])
