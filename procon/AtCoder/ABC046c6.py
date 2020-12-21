# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC
・二分探索
・昔と同じ愚直TLEをしたけど、すぐ内ループをにぶたんに切り替えてAC。
・公式解の切り上げだけでOKも言われればそうだけど、手早く通せるならこれでもいいよね。
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

N = INT()
cura, curb = MAP()
for i in range(N-1):
    a, b = MAP()
    # 比をそのままで、現在の票数より大きくする
    res = bisearch_min(0, INF, lambda m: a*m >= cura and b*m >= curb)
    cura, curb = a * res, b * res
ans = cura + curb
print(ans)
