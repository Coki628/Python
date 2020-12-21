# -*- coding: utf-8 -*-

"""
・速攻自力AC
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
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

ans = [0] * 6
for i in range(N):
    hi, low = map(float, input().split())
    if hi >= 35:
        ans[0] += 1
    if 35 > hi >= 30:
        ans[1] += 1
    if 30 > hi >= 25:
        ans[2] += 1
    if low >= 25:
        ans[3] += 1
    if hi >= 0 and low < 0:
        ans[4] += 1
    if hi < 0:
        ans[5] += 1
print(*ans)
