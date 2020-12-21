# -*- coding: utf-8 -*-

"""
・二分探索
・WA
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

N,K=MAP()
A=LIST()
S=[input() for i in range(N)]
S2=[S[A[i]-1] for i in range(K)]
S2.sort(reverse=True)
mxlen=len(S2[0])

def calc(idx):
    s=S2[0][:idx]
    cnt=0
    for i in range(N):
        if S[i].startswith(s):
            cnt+=1
    return cnt<=K

hi=mxlen
low=-1
while low+1<hi:
    mid=(hi+low)//2
    if calc(mid):
        hi=mid
    else:
        low=mid

s=S2[0][:hi]
ans=[]
for i in range(N):
    if S[i].startswith(s):
        ans.append(i+1)

if len(A)!=len(ans):
    print(-1)
    exit()

for i in range(K):
    if A[i]!=ans[i]:
        print(-1)
        exit()
print(s)
