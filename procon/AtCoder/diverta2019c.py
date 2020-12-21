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
S=['']*N
for i in range(N):
    S[i]=input()

enda=0
startb=0
both=0
cntab=0
for i in range(N):
    if S[i].startswith('B') and S[i].endswith('A'):
        both+=1
    elif S[i].startswith('B'):
        startb+=1
    elif S[i].endswith('A'):
        enda+=1
    cntab+=S[i].count('AB')

mnab=min(enda, startb)
cntab+=mnab
if both!=0:
    enda-=mnab
    startb-=mnab
    mxab=max(enda, startb)
    if mxab==0:
        cntab+=both-1
    else:
        if mxab<=both:
            cntab+=mxab
            both-=mxab
            cntab+=both
        else:
            cntab+=both
print(cntab)
