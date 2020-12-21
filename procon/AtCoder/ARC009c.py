# -*- coding: utf-8 -*-

"""
・TLE想定、部分点
・順列全探索
"""

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd
from itertools import permutations

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

def nCr(n, r):
    """ 事前テーブルなし組み合わせ簡易版 """
    from math import factorial

    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    return factorial(n) // (factorial(r) * factorial(n-r))

N, K = MAP()

def calc(k):
    cnt = 0
    for perm in permutations(range(k)):
        is_ok = True
        for i, p in enumerate(perm):
            if i == p:
                is_ok = False
        if is_ok:
            cnt += 1
    return cnt

print(nCr(N, K) * calc(K))
