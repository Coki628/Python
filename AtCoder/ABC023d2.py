# -*- coding: utf-8 -*-

"""
・蟻本演習3-1-2、自力AC！
・二分探索、式変形
・これは結構覚えてたから式変形分かった。
・類題と出くわしてもちゃんと解けるようになりたいね。
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

# 高度x以内で全て撃ち落とせるか
def calc(x):
    # 各風船に対する目標時間が定まる
    T = [0] * N
    for i in range(N):
        h, s = HS[i]
        # 式変形： t*s + h = x => t = (x-h) / s
        T[i] = (x-h) / s
    T.sort()
    for i, t in enumerate(T):
        # 1秒1つずつ撃ち落として、1つでも間に合わないものがあればNG
        if t < i:
            return False
    return True

N = INT()
HS = []
for _ in range(N):
    h, s = MAP()
    HS.append((h, s))

print(bisearch_min(0, 10**18, calc))
