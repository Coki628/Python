# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc034
・蟻本演習3-1-4
・二分探索、平均値の最大化
・目指す値を0地点とすることで、適切に優先順位がつけられる。
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

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索(小数用) """

    ok = mn
    ng = mx
    for _ in range(100):
        mid = (ok+ng) / 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

N, K = MAP()
WP = []
for _ in range(N):
    w, p = MAP()
    WP.append((w, p))

# 濃度mを達成可能か
def calc(m):
    # 濃度mにするためにその容器が得か損かの優先順位をつける
    A = [0] * N
    for i, (w, p) in enumerate(WP):
        A[i] = (w * (p-m), i)
    # 良い方からK個使って達成できればOK
    A.sort(reverse=True)
    a = b = 0
    for _, i in A[:K]:
        w, p = WP[i]
        a += w * (1 - p / 100)
        b += w * p / 100
    return b / (a+b) >= m / 100

ans = bisearch_max(0, 101, calc)
print(ans)
