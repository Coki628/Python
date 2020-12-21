# -*- coding: utf-8 -*-

"""
・自力AC
・k=1だと線形時間かかるから例外処理する、に気づくのにちょっと時間かかった。。
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

Q = INT()
for _ in range(Q):
    n, k = MAP()

    if k == 1:
        print(n-1)
        continue

    mx = 0
    i = 0
    powk = 1
    while mx < n:
        mx += powk
        powk *= k
        i += 1
    print(i-1)
