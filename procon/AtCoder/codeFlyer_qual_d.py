# -*- coding: utf-8 -*-

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product, accumulate
from operator import itemgetter, mul, xor
from copy import copy, deepcopy
from functools import reduce, partial
from fractions import Fraction
from string import ascii_lowercase, ascii_uppercase, digits

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def round(x): return int((x*2+1) // 2)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums): return reduce(lcm, nums, 1)
def gcd_list(nums): return reduce(gcd, nums, nums[0])
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

H,W=MAP()
N,M=MAP()

grid=[[None] for i in range(N)]
for i in range(N):
    grid[i]=list(input())
# L=[[INF]*M for i in range(N)]
# for i in range(N):
#     cnt=0
#     for j in range(1, M):
#         if grid[i][j-1]=='#':
#             cnt=1
#             L[i][j]=cnt
#         else:
#             cnt+=1
#             L[i][j]=cnt
# U=[[INF]*M for i in range(N)]
# for j in range(M):
#     cnt=0
#     for i in range(1, N):
#         if grid[i-1][j]=='#':
#             cnt=1
#             U[i][j]=cnt
#         else:
#             cnt+=1
#             U[i][j]=cnt
L=[[-1]*M for i in range(N)]
U=[[-1]*M for i in range(N)]
for i in range(N):
    for j in range(M):
        if j!=0:
            if grid[i][j-1]=='#':
                L[i][j]=1
            else:
                if L[i][j-1]!=-1:
                    L[i][j]=L[i][j-1]+1
        if i!=0:
            if grid[i-1][j]=='#':
                U[i][j]=1
            else:
                if U[i-1][j]!=-1:
                    U[i][j]=U[i-1][j]+1
R=[[-1]*M for i in range(N)]
D=[[-1]*M for i in range(N)]
for i in range(N-1, -1, -1):
    for j in range(M-1, -1, -1):
        if j!=M-1:
            if grid[i][j+1]=='#':
                R[i][j]=1
            else:
                if R[i][j+1]!=-1:
                    R[i][j]=R[i][j+1]+1
        if i!=N-1:
            if grid[i+1][j]=='#':
                D[i][j]=1
            else:
                if D[i+1][j]!=-1:
                    D[i][j]=D[i+1][j]+1
if M==1:
    for i in range(N):
        if grid[i][0]=='#':
            L[i][0]=R[i][0]=0
if N==1:
    for i in range(M):
        if grid[0][i]=='#':
            U[0][i]=D[0][i]=0

a=min(H-N, N*2)
b=min(W-M, M*2)
cnt=0
for i in range(N):
    for j in range(M):
        if grid[i][j]=='.':
            if (U[i][j]==-1 or D[i][j]==-1) and (L[i][j]==-1 or R[i][j]==-1):
                cnt+=1
            elif U[i][j]>a and L[i][j]>b:
                cnt+=1
            else:
                grid[i][j]='#'
                L[i][j]=R[i][j]=U[i][j]=D[i][j]=0
alrow=0
alcol=0
for i in range(N):
    for j in range(M):
        if L[i][j]!=-1 or R[i][j]!=-1:
            break
    else:
        alrow+=1
for j in range(M):
    for i in range(N):
        if U[i][j]!=-1 or D[i][j]!=-1:
            break
    else:
        alcol+=1
dup=alrow*alcol
cnt-=alrow*M+alcol*N-dup
cnt+=alrow*W+alcol*H-dup

if N==1 and M==1:
    if grid[0][0]=='.':
        print(0)
    else:
        print(H*W)
    exit()

print(H*W-cnt)
