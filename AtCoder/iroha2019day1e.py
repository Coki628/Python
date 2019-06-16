# -*- coding: utf-8 -*-

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

N,A,B=MAP()
D=[0]+sorted(LIST())+[N+1]

# 各記念日の間の日数
E=[]
for i in range(B+1):
    E.append(D[i+1]-D[i]-1)

# 記念日以外でデートが必要な日
F=[]
for e in E:
    F.append(e//A)

dsum=B+sum(F)
print(N-dsum)
