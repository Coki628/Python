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

N = INT()
X = LIST()
M = INT()
A = LIST()

fixed = [False] * 2021
pos = [0] * N
for i, x in enumerate(X):
    pos[i] = x
    fixed[x] = True

for a in A:
    a -= 1
    cur = pos[a]
    if cur == 2019 or fixed[cur+1]:
        continue
    pos[a] += 1
    fixed[cur] = False
    fixed[cur+1] = True

[print(p) for p in pos]
