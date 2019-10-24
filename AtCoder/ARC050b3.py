# -*- coding: utf-8 -*-

"""
・蟻本演習3-1-2、自力ならず。。これは悔しい。
・二分探索、式変形
・かなり覚えてたのに、自力で組んだやつはサンプル通ったのにWA。
・結局自分で解決できず、過去のやつ見て、同じことやっててすぐ気づいた。
・そもそも花が足りないケースは別途処理する必要があった。
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

# m個の花束が作れるか
def calc(m):
    # 必ず使う分
    r = R - m
    b = B - m
    # そもそも花が足りてない
    if r < 0 or b < 0:
        return False
    # 残りを使って、出来る限り作ってみて足りるか
    return r // (x-1) + b // (y-1) >= m

R, B = MAP()
x, y = MAP()

print(bisearch_max(0, 10**18, calc))
