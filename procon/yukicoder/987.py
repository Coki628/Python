# -*- coding: utf-8 -*-

import sys
from operator import add, mul

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

H, W = MAP()
B = list(input().split())
op, B = B[0], list(map(int, B[1:]))
A = LIST(H)

if op == '+':
    f = add
else:
    f = mul
ans = list2d(H, W, 0)
for i in range(H):
    for j in range(W):
        ans[i][j] = f(A[i], B[j])

for i in range(H):
    print(*ans[i])
