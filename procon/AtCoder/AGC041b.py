# -*- coding: utf-8 -*-

"""
・700点自力AC！嬉しいね。
・二分探索で、判定条件が複雑になるやつ。
・にぶたんだってことはネタバレしてたけど、それでも
　ちゃんと自分で考察して正しい条件作って通せたから良かった。
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

def check(m):
    # 元々P番目以内なのでOK確定
    if m >= N - P:
        return True
    # 自分に全投票した最善の状態
    target = A[m] + M
    # 残りの票数合計
    total = M * (V-1)
    # 絶対勝てないやつがP個以上あればNG確定
    r = bisect_right(A, target)
    if r <= N - P:
        return False
    # 元々自分より下に全投票(そうしても勝てるから)
    total -= M * m
    # P番目ギリギリに入れればいいので、関係ない上位層にも全投票
    total -= M * (P-1)
    # ボーダー上のものに、自分との差分ギリギリ(同率)まで投票
    for i in range(m+1, N-P+1):
        total -= target - A[i]
    # 票数が使い切れていればOK
    return total <= 0

N, M, V, P = MAP()
A = LIST()

A.sort()
res = bisearch_min(-1, N-1, check)
ans = N - res
print(ans)
