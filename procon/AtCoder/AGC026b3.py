# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2018/07/18/083000
・公式よりけんちょんさんのが分かりやすかった。
・自明に無理なケースを外したら、後はGCDとmodをうまく使う。
"""

import sys
from fractions import gcd

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
INF = 10 ** 19
MOD = 10 ** 9 + 7

def bisearch_min(mn, mx, func):
    """ 条件を満たす最小値を見つける二分探索 """

    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            # 下を探しに行く
            ok = mid
        else:
            # 上を探しに行く
            ng = mid
    return ok

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

for _ in range(INT()):
    a, b, c, d = MAP()

    # そもそも無理
    if a < b or d < b:
        No()
        continue
    
    # b個減ってd個増える繰り返しなので、このgより細かく値が動くことはない
    g = gcd(b, d)

    # cより大きくてb未満が存在するかどうか
    mn = bisearch_min(-INF, INF, lambda m: a%g + g*m > c)
    mx = bisearch_max(-INF, INF, lambda m: a%g + g*m < b)
    if mn > mx:
        Yes()
    else:
        No()
