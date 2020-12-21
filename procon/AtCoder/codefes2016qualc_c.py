# -*- coding: utf-8 -*-

"""
・400点自力AC！
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
T=LIST()
A=LIST()

# 累積MAXが変化するところは確定と言える
Hs1=[0]*N
Hs1[0]=T[0]
fixed=[False]*N
fixed[0]=True
for i in range(1, N):
    if T[i-1]!=T[i]:
        Hs1[i]=T[i]
        fixed[i]=True
Hs2=[0]*N
Hs2[-1]=A[-1]
fixed[-1]=True
for i in range(N-2, -1, -1):
    if A[i+1]!=A[i]:
        Hs2[i]=A[i]
        fixed[i]=True

for i in range(N):
    # 両者の確定箇所について矛盾がないか確認
    if Hs1[i]!=0 and Hs2[i]!=0 and Hs1[i]!=Hs2[i]:
        print(0)
        exit()
    # 一方が確定している場合の、他方の累積MAXとの確認
    if Hs1[i]!=0:
        if Hs1[i]>A[i]:
            print(0)
            exit()
    if Hs2[i]!=0:
        if Hs2[i]>T[i]:
            print(0)
            exit()
ans=1
for i in range(N):
    if not fixed[i]:
        # 未確定の場所について、低い方の数だけ可能性がある
        ans=(ans*min(T[i], A[i]))%MOD
print(ans)
