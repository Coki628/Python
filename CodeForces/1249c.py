# -*- coding: utf-8 -*-

"""
・自力AC！
・ダブリングで2冪から位置を探す時の要領で、大きい方から3冪での位置を探す。
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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

MAX = 0
for i in range(40):
    MAX += 3 ** i

for _ in range(INT()):
    N = INT()
    ans = MAX
    for i in range(39, -1, -1):
        if ans - 3**i >= N:
            ans -= 3**i
    print(ans)
