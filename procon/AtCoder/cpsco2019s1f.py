# -*- coding: utf-8 -*-

"""
・自力AC！
・二分探索、式変形、優先度付きキュー
・式変形も優先度付きキューの区間シミュレーションも、自力ダメな時多いからちゃんと解けて嬉しい。
・特に優先度付きキューの、使えるようになったものから順次追加する、
　は、他の問題でも結構見たことあって解説ACしてた気がするから、自分でちゃんと使えて良かった。
・計算量は2万のNににぶたんと優先度キューでlog2つ付いて、pythonAC1.5秒、pypyAC0.8秒。
"""

import sys
from heapq import heappush, heappop

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

def bisearch_max(mn, mx, func):
    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

# 満足度mが可能か
def check(m):
    # 満足度がmの時に各果物が食べられる日程の範囲を取る
    segs = [[] for i in range(N+1)]
    for t, a, b in TAB:
        # a - b*day >= m から式変形
        day = (a-m) // b
        # 当日でも無理なものが1つでもあればNG
        if day < 0:
            return False
        l = max(t-day, 1)
        r = min(t+day, N)
        # 左端毎にまとめておく
        segs[l].append(r)

    que = []
    for cur in range(1, N+1):
        # 現在日で食べてもよくなる果物をキューに追加
        for r in segs[cur]:
            heappush(que, r)
        # 何も食べれない日があったらNG
        if not que:
            return False
        # 期限(右端)が一番短いものから優先して取る
        r = heappop(que)
        # これが既に期限切れだったらNG
        if r < cur:
            return False
    # 最終日までうまくいけばOK
    return True

N = INT()
TAB = []
for i in range(N):
    t, a, b = MAP()
    TAB.append((t, a, b))

res = bisearch_max(-INF, INF, check)
print(res)
