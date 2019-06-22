# -*- coding: utf-8 -*-

import sys
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

N=INT()
H=[]
for i in range(N):
    H.append([i, INT()])

H.sort(key=itemgetter(1))
h=H[0][1]
cnt=1
H[0].append(cnt)
for i in range(1, N):
    if H[i][1]!=h:
        cnt+=1
        h=H[i][1]
    H[i].append(cnt)

H.sort(key=itemgetter(0))
for i in range(N):
    print(H[i][2])
