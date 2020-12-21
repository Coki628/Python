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

N=INT()
S=input()

sm1=[0]*N
prev=False
if S[0]=='#':
    sm1[0]=1
    prev=True
for i in range(1, N):
    if S[i]=='#':
        if prev:
            sm1[i]=sm1[i-1]+1
        else:
            sm1[i]=1
            prev=True
    else:
        prev=False
sm2=[0]*N
prev=False
if S[-1]=='.':
    sm2[-1]=1
    prev=True
for i in range(N-1, 0, -1):
    if S[i-1]=='.':
        if prev:
            sm2[i-1]=sm2[i]+1
        else:
            sm2[i-1]=1
            prev=True
    else:
        prev=False

S2=list(S)
cnt1=0
for i in range(1, N):
    if S2[i-1]=='#' and S2[i]=='.':
        if sm1[i-1]>=sm2[i]:
            S2[i]='#'
            cnt1+=1
for i in range(N-1, 0, -1):
    if S2[i-1]=='#' and S2[i]=='.':
        if sm1[i-1]<sm2[i]:
            S2[i-1]='.'
            cnt1+=1
S3=list(S)
cnt2=0
for i in range(1, N):
    if S3[i-1]=='#' and S3[i]=='.':
        if sm1[i-1]>sm2[i]:
            S3[i]='#'
            cnt2+=1
for i in range(N-1, 0, -1):
    if S3[i-1]=='#' and S3[i]=='.':
        if sm1[i-1]<=sm2[i]:
            S3[i-1]='.'
            cnt2+=1

print(min(cnt1, cnt2))
