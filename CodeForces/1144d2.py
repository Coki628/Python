# -*- coding: utf-8 -*-

"""
・完全にケアレスミス。必要な値の更新してなかった…。
"""

import sys
from collections import Counter

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

C=Counter(A)
C=C.most_common()
base=C[0][0]

bases=[]
for i in range(N):
    if A[i]==base:
        bases.append(i)

ans=[]
for i in range(len(bases)):
    for j in range(bases[i]-1, -1, -1):
        if A[j]==base:
            break
        else:
            if A[j]<base:
                ans.append((1, j+1, j+2))
            else:
                ans.append((2, j+1, j+2))
            A[j]=base
    for j in range(bases[i]+1, N):
        if A[j]==base:
            break
        else:
            if A[j]<base:
                ans.append((1, j+1, j))
            else:
                ans.append((2, j+1, j))
            A[j]=base
print(len(ans))
for t, i, j in ans:
    print(t, i, j)
