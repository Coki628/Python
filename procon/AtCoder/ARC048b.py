# -*- coding: utf-8 -*-

"""
・さくっと自力AC
"""

import sys
from collections import defaultdict
from bisect import bisect_left, bisect_right

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
D = defaultdict(lambda: [0]*3)
RH = []
ratings = []
for i in range(N):
    r, h = MAP()
    RH.append((r, h-1))
    ratings.append(r)
    D[r][h-1] += 1

ratings.sort()
for r, h in RH:
    # レーティング確定分
    win = bisect_left(ratings, r)
    lose = N - bisect_right(ratings, r)
    # じゃんけん勝敗分
    win += D[r][(h+1)%3]
    lose += D[r][(h-1)%3]
    draw = D[r][h] - 1
    print(win, lose, draw)
