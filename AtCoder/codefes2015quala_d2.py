# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/codefestival2015quala
・蟻本演習3-1-3
・二分探索、最小値の最大化(最大値の最小化)
・チェック関数内でDPっぽくしたり、別のにぶたんしたり。
・計算量10万のMと10億のNでMlogN^2=10万*30^2=9000万、、log2つだときつい。pypyでもTLE。
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
INF = float('inf')
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

T = 0
L = 0
def check(t):
    global T
    global L
    T = t
    # dp[i] := i人目の整備士が、最大何両目まで点検できるか
    dp = [0] * (M+1)
    dp[0] = 0
    for i in range(M):
        # 左に絶対行く必要のある距離L
        L = max(A[i] - dp[i] - 1, 0)
        # これがT以内でなければこの時点でNG
        if L > T:
            return False
        # Lを踏まえて、右に行ける最大距離R
        R = bisearch_max(0, N, check2)
        dp[i+1] = A[i] + R
    # 最後の1人までやってNに届いていればOK
    return dp[M] >= N

def check2(r):
    # 左→右、右→左、のどちらかがT以内で行けるかどうか
    return min(L*2+r, L+r*2) <= T

N, M = MAP()
A = LIST(M)

# 整備士が真ん中に1人だとN時間でも全部回れないので、上限は大きく取る
print(bisearch_min(-1, N*2, check))
