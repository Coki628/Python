# -*- coding: utf-8 -*-

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd
from functools import reduce
from collections import Counter
from operator import itemgetter

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

def lcm(x, y): return (x * y) // gcd(x, y)
def gcd_list(li): return reduce(gcd, li, 0)

N=INT()

A=[]
C=Counter()
for i in range(N):
    a,b=MAP()
    A.append(lcm(a, b))
    C[a]+=1
    C[b]+=1

if A.count(A[0])==len(A):
    C=list(C.items())
    C.sort(key=itemgetter(0), reverse=True)
    C.sort(key=itemgetter(1), reverse=True)
    print(C[0][0])
else:
    print(gcd_list(A))
