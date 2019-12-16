# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・3重ループを2重+logに落とす。
・にぶたん内でにぶたん
・100万にlog2つ乗ってpypyでもつらすぎるのでC++で。
・惜しくも自力ACならず。。
・同値がたくさんある時、出力をKより多く出してた。
・それだけ直したら無事C++AC1.2秒。
"""

import sys
from bisect import bisect_left

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

X, Y, Z, K = MAP()
A = LIST()
B = LIST()
C = sorted(LIST())

# K個以上の3つ組が達成できるような最大の値mを探す
def check(m):
    sm = 0
    for a in A:
        for b in B:
            cnt = Z - bisect_left(C, m-a-b)
            sm += cnt
    return sm >= K

res = bisearch_max(0, 10**18, check)

ans = []
for a in A:
    for b in B:
        idx = bisect_left(C, res-a-b)
        for i in range(Z-1, idx-1, -1):
            c = C[i]
            ans.append(a+b+c)
ans.sort(reverse=1)
# K個まで出力
for i in range(K):
    print(ans[i])
