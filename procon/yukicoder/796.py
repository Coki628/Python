# -*- coding: utf-8 -*-

"""
・自力AC
・数列構築
・総乗の方は3が1つでもあればよくて、総和の方はmod 3で1になるように。
　あとは上の2条件に合うように小さい値で作ればOK。
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

N = INT()

if N % 3 == 2:
    ans = [3] + [1] * (N-1)
elif N % 3 == 1:
    ans = [3] + [1] * (N-2) + [2]
else:
    ans = [3] + [1] * (N-2) + [3]
print(*ans)
