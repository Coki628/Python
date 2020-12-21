# -*- coding: utf-8 -*-

"""
・300点自力AC
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

T=input()
N=len(T)

# 2パターン作ってどっちか良い方
S1=['']*N
for i in range(N):
    if i%2==0 and T[i]=='?':
        S1[i]='2'
    elif i%2==1 and T[i]=='?':
        S1[i]='5'
    else:
        S1[i]=T[i]
S2=['']*N
for i in range(N):
    if i%2==0 and T[i]=='?':
        S2[i]='5'
    elif i%2==1 and T[i]=='?':
        S2[i]='2'
    else:
        S2[i]=T[i]

mx=0
for i in range(2):
    cnt1=cnt2=0
    for j in range(i, N, 2):
        if j+1<N:
            if S1[j]=='2' and S1[j+1]=='5':
                cnt1+=2
            else:
                cnt1=0
            if S2[j]=='2' and S2[j+1]=='5':
                cnt2+=2
            else:
                cnt2=0
        mx=max(mx, cnt1, cnt2)
print(mx)
