# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・二分探索
・ちょっと迷走したけど、考察整ったらすぐ自力AC。
"""

import sys
from bisect import bisect_left, bisect_right

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

N, M, Q = MAP()
A = [0] * N
for i in range(N):
    A[i] = INT()
B = [0] * M
for i in range(M):
    B[i] = INT()

A.sort()
B.sort()
A = [-INF] + A + [INF]
B = [-INF] + B + [INF]

for i in range(Q):
    q = INT()
    
    # q以下で最大
    al = bisect_right(A, q) - 1
    bl = bisect_right(B, q) - 1
    # q以上で最小
    ar = bisect_left(A, q)
    br = bisect_left(B, q)

    # 両方左
    a = max(q-A[al], q-B[bl])
    # 両方右
    b = max(A[ar]-q, B[br]-q)
    # 左の神社、右の寺(近い方に先に行ってから遠い方を目指すから、近い方を*2)
    c = min(q-A[al], B[br]-q) * 2 + max(q-A[al], B[br]-q)
    # 右の神社、左の寺
    d = min(A[ar]-q, q-B[bl]) * 2 + max(A[ar]-q, q-B[bl])

    ans = min(a, b, c, d)
    print(ans)
