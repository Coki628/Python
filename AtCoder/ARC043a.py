# -*- coding: utf-8 -*-

"""
・式変形
・平均Aと差Bについて連立方程式を立てると、
　2つの式からP,Qを出せる。
・なんとか自力で辿り着いたけど、やっぱこれ系解くの遅すぎるわぁ。。
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

N,A,B=MAP()
S=[INT() for i in range(N)]

S.sort()
Smn=S[0]
Smx=S[-1]
Ssm=sum(S)

if Smn==Smx:
    print(-1)
    exit()

P=B/(Smx-Smn)
Q=(A*N-Ssm*P)/N

print(P, Q)
