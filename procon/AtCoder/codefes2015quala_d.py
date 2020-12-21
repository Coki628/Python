# -*- coding: utf-8 -*-

"""
・蟻本演習3-1-3、自力WA
・二分探索、最小値の最大化(最大値の最小化)
・判定内の処理が色々ある。
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

def check(x):
    done = 0
    for i in range(M):
        a = A[i]
        # aの初期位置と塗りたい範囲の関係で場合分け
        if a < done:
            if abs(done+x-a) >= x:
                return False
            else:
                done = min(done + x, A[i+1])
        elif done <= a < done + x:
            done = min(done + x, A[i+1])
        elif done + x <= a:
            if abs(done-a) >= x:
                return False
            else:
                done = a
    return True

N, M = MAP()
A = [a-1 for a in LIST(M)] + [INF]

print(bisearch_min(0, N, check) - 1)
