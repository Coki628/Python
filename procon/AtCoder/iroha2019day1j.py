# -*- coding: utf-8 -*-

"""
・調査用
"""

import sys
from itertools import product
from collections import Counter

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

L=[[['0'], ['1']]]
for i in range(1, N+1):
    A=[]
    B=[]
    for prod in product(['0', '1'], repeat=i):
        A.append(list(prod) + list(reversed(prod)))
        B.append(list(prod) + ['0'] + list(reversed(prod)))
        B.append(list(prod) + ['1'] + list(reversed(prod)))
    L.append(A)
    L.append(B)

def tentocnt(S):
    cnt=0
    for i, s in enumerate(S):
        if s=='1':
            cnt+=S[i+1:].count('0')
    return cnt

cnts=[]
for l in L:
    c=Counter()
    for s in l:
        cnt=tentocnt(''.join(s))
        c[cnt]+=1
    cnts.append(c)

print()
