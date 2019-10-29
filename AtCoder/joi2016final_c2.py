# -*- coding: utf-8 -*-

"""
参考：https://www.ioi-jp.org/joi/2016/2017-ho/2017-ho-t3-review.pdf
・蟻本演習3-1-3
・二分探索、最小値の最大化(最大値の最小化)
・にぶたん内のGreedyパートが思いつきづらい。。
・計算量HWとlogで400万*30くらい=1億2000万、、しかも回転させて計4回。
　よく通ったな。まあ途中で打ち切ったりで全部は回してないだろうけど。。
・pypyAC3.0秒(制約4秒)
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

def rot90(grid, H, W):
    res = list2d(W, H, 0)
    for i in range(H):
        for j in range(W):
            res[j][H-i-1] = grid[i][j]
    return res

H, W = MAP()
grid = [None] * H
mx, mn = 0, INF
for i in range(H):
    grid[i] = LIST()
    # 全体の最大最小を持っておいて、片方が最大、反対が最小を持つと決めておく
    mx = max(mx, max(grid[i]))
    mn = min(mn, min(grid[i]))
# グリッドは90°回転させて4つ作ると実装が楽
grid2 = rot90(grid, H, W)
grid3 = rot90(grid2, W, H)
grid4 = rot90(grid3, H, W)

def check(k):
    return (check2(k, grid, H, W)
            or check2(k, grid2, W, H)
            or check2(k, grid3, H, W)
            or check2(k, grid4, W, H)
    )

# 答えをkとしてうまくいくか
def check2(k, grid, H, W):
    # 青が最小値、赤が最大値を持つとして、両方の最大距離をk以内にできるか
    mx_limit = mn + k
    mn_limit = mx - k
    # 境界位置
    cur = 0
    for i in range(H):
        # 青には属せないマスで最も右の場所を探す
        for j in range(W):
            if grid[i][j] > mx_limit:
                cur = max(cur, j+1)
        # その場所よりも左に赤に属せないマスがあったらNG
        for j in range(cur-1, -1, -1):
            if grid[i][j] < mn_limit:
                return False
    return True

print(bisearch_min(-1, mx-mn, check))
