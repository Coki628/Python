# -*- coding: utf-8 -*-

"""
参考：http://arc050.contest.atcoder.jp/data/arc/050/review.pdf
　　　https://atcoder.jp/contests/arc050/submissions/6865909
・二分探索
・ちょっとそれは考えたけど、判定する計算がうまく作れなかった。。
・しかも解説見て実装したらWA、そもそも花が足りてないやつ弾けてなかった。
・この辺は前提知識は足りてるんだから解けて欲しいよなー。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

R, B = MAP()
x, y = MAP()

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

# K個の花束を作れるかどうか
def calc(K):
    # 前提として最低限必要な花が足りてるかチェック
    if R < K or B < K:
        return False
    # 絶対に必要な1本を先に減らしておくことで、見通しが良くなる
    r = R - K
    b = B - K
    # 反対側に使われる1本を考えずに、残りの必要本数を計算できる
    redset = r // (x-1)
    blueset = b // (y-1)
    return redset + blueset >= K

mx = R + B + 1
res = bisearch_max(0, mx, calc)
print(res)
