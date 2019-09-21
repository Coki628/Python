# -*- coding: utf-8 -*-

"""
・400点さくっと自力AC！
・二分探索
・ちゃんと正しい視点でにぶたんできて良かった。
"""

import sys
from bisect import bisect_right

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

N = INT()
A = LIST()
A.sort()

# m番目の大きさのやつが、最後の1匹になれるか
def calc(m):
    # 番兵代わり
    if m == -1:
        return False
    # まず自分以下は全部取り込む
    a = sum(A[:m+1])
    # 自分の2倍まではいける
    idx = bisect_right(A, a*2)
    a += sum(A[m+1:idx])
    # 全て取り込んだ後で、その先に行ければ行く
    for i in range(idx, N):
        b = A[i]
        if a*2 >= b:
            a += b
        else:
            # 途中で取り込めなくなったらNG
            return False
    # 最後まで行けるなら、最後の1匹になれる
    return True

print(N-bisearch_min(-1, N-1, calc))
