# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・風の強い月を前もって調べておいて、後は普通にシミュレーション。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, M, p, q = MAP()

windy = [0] * 12
for i in range(p-1, p+q-1):
    windy[i] = 1

cur = 0
ans = 0
while N > 0:
    if windy[cur%12]:
        N -= 2 * M
    else:
        N -= M
    cur += 1
    ans += 1
print(ans)
