# -*- coding: utf-8 -*-

"""
・ちょっと考えたけど自力AC！
・前計算、二分探索
・計算量は50万のNと50万クエリにlog、pypyAC2.2秒。(制約5秒)
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

# x番目の人の時刻t時点での位置
def get_pos(x, t):
    return A[x] + max(t - acc[x], 0)

def check1(m):
    return get_pos(m, t) <= r

def check2(m):
    return get_pos(m, t) >= l

N, Q = MAP()
D = [1] + LIST(N)

# A[i] := i番目の人の初期位置
A = [0] * (N+1)
for i in range(N+1):
    A[i] = -i

# acc[i] := i番目の人が動き出すまでにかかる時間
acc = [0] * (N+1)
for i in range(N+1):
    acc[i] = D[i] - 1
acc = list(accumulate(acc))

ans = [''] * Q
for i in range(Q):
    t, l, r = MAP()
    
    # 条件を満たす右端の最小
    res1 = bisearch_min(-1, N+1, check1)
    # 条件を満たす左端の最大
    res2 = bisearch_max(-1, N+1, check2)
    if res2 < res1:
        ans[i] = '0'
    else:
        ans[i] = str(res2 - res1 + 1)
print('\n'.join(ans))
