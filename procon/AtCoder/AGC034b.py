# -*- coding: utf-8 -*-

"""
・再帰的に戻って確認
・これはTLE
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

S=list(input())
N=len(S)

def rec(cur):
    global cnt
    if cur+2<N:
        if S[cur]=='A' and S[cur+1]=='B' and S[cur+2]=='C':
            S[cur]='B'
            S[cur+1]='C'
            S[cur+2]='A'
            cnt+=1
            rec(cur+2)
i=N-1
cnt=0
while i>=0:
    if i-2>=0:
        if S[i-2]=='A' and S[i-1]=='B' and S[i]=='C':
            S[i-2]='B'
            S[i-1]='C'
            S[i]='A'
            cnt+=1
            rec(i)
    i-=1
print(cnt)
