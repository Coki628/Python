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

S=input()
N=len(S)

lcnt=S.count('(')
rcnt=S.count(')')
changed=['0']*N

T=list(S)
if lcnt==rcnt:
    l=r=0
    for i in range(N):
        if S[i]=='(':
            l+=1
        else:
            r+=1
        if r>l:
            T[i]='('
            l+=1
            r-=1
            changed[i]='1'
    l=r=0
    for i in range(N-1, -1, -1):
        if S[i]=='(':
            l+=1
        else:
            r+=1
        if r<l:
            T[i]=')'
            r+=1
            l-=1
            changed[i]='1'
elif lcnt>rcnt:
    l=r=0
    for i in range(N-1, -1, -1):
        if S[i]=='(':
            l+=1
        else:
            r+=1
        if r<l:
            T[i]=')'
            r+=1
            l-=1
            changed[i]='1'
else:
    gap=rcnt-lcnt
    for i in range(gap):
        idx=S.find(')')
        T[idx]='('
        changed[idx]='1'

changed=''.join(changed)
ans=changed.count('1')
ans+=changed.rfind('1')

print(ans)
