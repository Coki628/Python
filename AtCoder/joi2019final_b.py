# -*- coding: utf-8 -*-

"""
・自力AC！
・累積和で前計算して二分探索
・3つある何かを決める時に真ん中を固定して考えるとうまくいくやつ。
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

N, K = MAP()
S = '*' + input()

accj = [0] * (N+1)
acci = [0] * (N+1)
idxo = []
for i in range(1, N+1):
    if S[i] == 'J':
        accj[i] = 1
    elif S[i] == 'I':
        acci[i] = 1
    else:
        idxo.append(i)
accj = list(accumulate(accj))
acci = list(accumulate(acci))

# Oがそもそも足りなくてNG
if len(idxo) < K:
    print(-1)
    exit()

ans = INF
for i in range(len(idxo)-K+1):
    # K個のOを含む範囲l,r
    l, r = idxo[i], idxo[i+K-1]
    # lより手前でK個のJが取れる最大indexを二分探索
    idxj = bisearch_max(0, N+1, lambda m: accj[l] - accj[m-1] >= K)
    if idxj == 0:
        continue
    # rより先でK個のIが取れる最小indexを二分探索
    idxi = bisearch_min(0, N+1, lambda m: acci[m] - acci[r] >= K)
    if idxi == N+1:
        continue
    # K個のJOIを含む範囲lnからJOIに使うK*3個を除けば、必要な操作回数になる
    ln = idxi - idxj + 1
    ans = min(ans, ln - K*3)
if ans == INF:
    print(-1)
else:
    print(ans)
