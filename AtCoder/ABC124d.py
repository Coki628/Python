# -*- coding: utf-8 -*-

"""
・400点なんとか自力AC
・累積和
"""

import sys
from itertools import accumulate

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
S=input()

if N==1:
    print(1)
    exit()

L=[]
cnt0=0
if S[0]=='0':
    L.append(0)
    cur=0
else:
    cur=1
cnt=0
for i in range(N):
    if cur==0:
        if S[i]=='0':
            cnt+=1
        else:
            L.append(cnt)
            cnt=1
            cur=1
            cnt0+=1
    else:
        if S[i]=='1':
            cnt+=1
        else:
            L.append(cnt)
            cnt=1
            cur=0
else:
    if cnt!=0:
        if cur==0:
            L.append(cnt)
            cnt0+=1
        else:
            L.append(cnt)
if S[-1]=='0':
    L.append(0)

acc=[0]+list(accumulate(L))
mx=0
if cnt0<K:
    K=cnt0
for i in range(0, len(acc), 2):
    if i+(K*2+1)<len(acc):
        mx=max(mx, acc[i+(K*2+1)]-acc[i])
print(mx)
