# -*- coding: utf-8 -*-

"""
・numpy実装
・10万数列の二分探索をsearchsortedでまとめて処理。
・pythonAC0.9秒。さすが、pypyより速い。
"""

import sys
from bisect import bisect_left
from itertools import accumulate

import numpy as np

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

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索 """

    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

# 右手と左手(a,b)の和がx以上となる組の数がM以上あるか
def calc(x):
    # 数列Aの二分探索をまとめて処理
    res = N - np.searchsorted(A, x - A)
    return res.sum() >= M

N, M = MAP()
A = np.array(LIST(), dtype=np.int64)

A.sort()
res = bisearch_max(0, 10**10 + 1, calc)

ans = m = 0
acc = [0] + list(A.cumsum())
for a in A:
    idx = bisect_left(A, res - a)
    cnt = N - idx
    # 幸福度
    ans += a * cnt + (acc[N] - acc[idx])
    # 握手の回数
    m += cnt

# 同値でMを超えた分の調整
ans -= res * (m - M)
print(ans)
