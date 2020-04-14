# -*- coding: utf-8 -*-

"""
・ここまでは自力AC。
・種類毎に累積和、区切り位置を全探索
・こっちの制約弱い方は2重ループできるので、
　3つのブロックの区切り位置2つを両方全探索できる。
"""

import sys
from itertools import accumulate

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    A = [0] + [a-1 for a in LIST()]

    acc = list2d(26, N+1, 0)
    for i, a in enumerate(A[1:], 1):
        acc[a][i] = 1
    for a in range(26):
        acc[a] = list(accumulate(acc[a]))

    ans = 0
    for i in range(N+1):
        for j in range(i, N+1):
            y = 0
            for a in range(26):
                y = max(y, acc[a][j] - acc[a][i])
            x = 0
            for a in range(26):
                x = max(x, min(acc[a][i], acc[a][N] - acc[a][j]))
            ans = max(ans, x*2 + y)
    print(ans)
