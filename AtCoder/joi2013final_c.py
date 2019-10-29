# -*- coding: utf-8 -*-

"""
・蟻本演習3-1-3、自力AC！
・二分探索、最小値の最大化(最大値の最小化)
・にぶたん内でにぶたん系
・計算量はNlogNlogmax(A)=10万*30*17くらい=5100万
・log2つ乗るとやっぱきつくて、python,pypyTLE。。
・C++でさすがの0.3秒AC。
"""

import sys
from itertools import accumulate
from bisect import bisect_left

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
INF = float('inf')
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

N = INT()
A = LIST(N)

# 累積和を2周分作る
acc = list(accumulate(A+A))
total = sum(A)
offset = 0
def check(m):
    # 1つ目の切れ込み
    idx1 = bisect_left(acc[i:i+N], m+offset)
    a1 = acc[i+idx1] - offset
    # 2つ目の切れ込み
    idx2 = bisect_left(acc[i:i+N], m+a1+offset)
    a2 = acc[i+idx2] - offset
    # 最後のピースのサイズがm以上あればOK
    return total - a2 >= m

ans = 0
for i in range(N):
    # 始点iを全箇所試す
    res = bisearch_max(0, total//2, check)
    ans = max(ans, res)
    offset += A[i]
print(ans)
