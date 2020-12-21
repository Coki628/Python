# -*- coding: utf-8 -*-

"""
・300点速攻自力AC
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

S=input()
N=len(S)

T1=''
for i in range(N):
    if i%2==0:
        T1+='0'
    else:
        T1+='1'
T2=''
for i in range(N):
    if i%2==0:
        T2+='1'
    else:
        T2+='0'

cnt1=0
for i in range(N):
    if S[i]!=T1[i]:
        cnt1+=1
cnt2=0
for i in range(N):
    if S[i]!=T2[i]:
        cnt2+=1
print(min(cnt1, cnt2))
