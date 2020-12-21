# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc029
・ビット全探索
・計算量O(2^N)
"""

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
A = [INT() for i in range(N)]

M = sum(A)
ans = INF
for i in range(1<<N):
    sm = 0
    for j in range(N):
        if i>>j&1:
            sm += A[j]
    if sm >= ceil(M, 2):
        ans = min(ans, sm)
print(ans)
