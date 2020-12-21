# -*- coding: utf-8 -*-

"""
・400点自力AC
・まあ、何とかなったけど、RLってとこに集約されるの気付くの遅かったー。
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

A=[]
ecnt=1
ocnt=idx=0
for i in range(1, N):
    if S[i-1]==S[i]=='R' or S[i-1]==S[i]=='L':
        if i%2==0:
            ecnt+=1
        else:
            ocnt+=1
    elif S[i-1]=='R' and S[i]=='L':
        idx=i
        if i%2==0:
            ecnt+=1
        else:
            ocnt+=1
    elif S[i-1]=='L' and S[i]=='R':
        A.append((ecnt, ocnt, idx))
        if i%2==0:
            ecnt=1
            ocnt=0
        else:
            ocnt=1
            ecnt=0
        idx=0
else:
    A.append((ecnt, ocnt, idx))

ans=[0]*N
for a in A:
    ecnt,ocnt,idx=a
    if idx%2==0:
        ans[idx]=ecnt
        ans[idx-1]=ocnt
    else:
        ans[idx]=ocnt
        ans[idx-1]=ecnt
print(*ans)
