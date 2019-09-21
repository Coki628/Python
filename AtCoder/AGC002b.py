# -*- coding: utf-8 -*-

"""
・400自力AC
"""

import sys

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
INF = float('inf')
MOD = 10 ** 9 + 7

N, M = MAP()

has_red = [False] * N
has_red[0] = True
box = [1] * N
for _ in range(M):
    x, y = MAP()
    x -= 1
    y -= 1
    box[x] -= 1
    box[y] += 1
    if has_red[x]:
        has_red[y] = True
    if box[x] == 0:
        has_red[x] = False

ans = 0
for i in range(N):
    if box[i] != 0 and has_red[i]:
        ans += 1
print(ans)
