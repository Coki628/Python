# -*- coding: utf-8 -*-

"""
・400点自力AC
・式変形(詳細は自分ノート参照)
　ある頂点をXとすると最終的に、2X=各辺の値を+-してったやつ、になる。
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

tmp=0
for i in range(N):
    if i%2==0:
        tmp+=A[i]
    else:
        tmp-=A[i]

ans=[0]*N
ans[0]=tmp
for i in range(1, N):
    ans[i]=(A[i-1]-ans[i-1]//2)*2

print(*ans)
