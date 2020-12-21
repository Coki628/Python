# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/70860
・二分探索。最近たくさんやってる系だしこれは自力解きたかったなー。
・判定内で都度、3つのグループに分類する。
・計算量は20万のNににぶたんのlog+全てではないけどソートもするのでちょっと重め。
・pythonTLEでpypyAC2.2秒(制約3秒)
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

# 中央値mを達成可能か
def check(m):
    A = []
    cnt = k = 0
    # 従業員を3つのグループに分ける
    for l, r in LR:
        # 絶対mに届かない
        if r < m:
            k += l
        # 絶対m以上になる
        elif m <= l:
            cnt += 1
            k += l
        # 個別にどうするか判定する
        else:
            A.append((l, r))
            k += l
    # 最低限時点で予算オーバー
    if k > K:
        return False
    # この先全員合わせても足りない
    if cnt+len(A) < midcnt:
        return False
    # 最低限時点で目標達成
    if cnt >= midcnt and k <= K:
        return True
    # 残りのmまでの額が少ない方からmにしていく
    A.sort(reverse=True)
    for l, r in A:
        cnt += 1
        k += m - l
        # 予算オーバー
        if k > K:
            return False
        # 目標達成
        if cnt >= midcnt and k <= K:
            return True
    return False

for _ in range(INT()):
    N, K = MAP()
    midcnt = N//2 + 1
    LR = []
    for i in range(N):
        l, r = MAP()
        LR.append((l, r))
    res = bisearch_max(0, 10**9+1, check)
    print(res)
