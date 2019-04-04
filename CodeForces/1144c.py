# -*- coding: utf-8 -*-

import sys
from collections import Counter, deque

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
A=LIST()

C=Counter(A)
asc=deque()
desc=deque()
for k, v in sorted(C.items()):
    if v>=3:
        NO()
        exit()
    elif v==2:
        asc.append(k)
        desc.appendleft(k)
    elif v==1:
        asc.append(k)
YES()
print(len(asc))
print(*asc)
print(len(desc))
print(*desc)
