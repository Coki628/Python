# -*- coding: utf-8 -*-

"""
・300点自力AC
・とはいえ貪欲でいけることに確証はなかったけど。
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

N,C=MAP()
L=[INT() for i in range(N)]

L.sort()
i=0
j=N-1
cnt=N
while i<j:
    if L[i]+L[j]+1<=C:
        cnt-=1
        i+=1
    j-=1
print(cnt)
