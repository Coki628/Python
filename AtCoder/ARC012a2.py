# -*- coding: utf-8 -*-

import sys
from itertools import permutations

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

N, M, L = MAP()
P, Q, R = MAP()

mx = 0
# 荷物の向きを6通り全部試す
for perm in permutations([P, Q, R]):
    p, q, r = perm
    res = (N//p) * (M//q) * (L//r)
    mx = max(mx, res)
print(mx)
