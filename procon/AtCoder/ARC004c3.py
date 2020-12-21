# -*- coding: utf-8 -*-

"""
参考：http://techtipshoge.blogspot.com/2012/06/atcoder-regular-contest-004-c.html
・式変形、二分探索
・ブログではみんな式変形を工夫して、計算O(1)で必要な範囲を出していた。
・それを自分で導ける気はしなかったけど、この二分探索ならほぼ自力で組めた。
・基本は自力WAと中身同じで、上限と下限をそれぞれ別の二分探索で探しに行く。
・速度もAC0.1秒以下で問題ない。まあ計算量、計算O(1)にlogついただけだからね。
"""

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd

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

X, Y = map(int, input().split('/'))

# 約分しておく
_gcd = gcd(X, Y)
if _gcd != 1:
    X //= _gcd
    Y //= _gcd

orgX, orgY = X, Y

def calc1(m):
    X = orgX * m
    Y = orgY * m
    # 1からYの等差数列の和
    sm = (Y+1)*Y // 2
    M = sm - X
    return M <= Y

def calc2(m):
    X = orgX * m
    Y = orgY * m
    sm = (Y+1)*Y // 2
    M = sm - X
    return M > 0

ans = []
def calc3(m):
    X = orgX * m
    Y = orgY * m
    sm = (Y+1)*Y // 2
    # 範囲内なのは探索済なので答えに加える
    M = sm - X
    N = Y
    ans.append((N, M))

# 上限の確定
r = bisearch_max(0, 10**18+1, calc1)
# 下限の確定
l = bisearch_min(-1, 10**18, calc2)
# 候補となった範囲だけ探しに行く
for i in range(l, r+1):
    calc3(i)
if ans:
    [print(*a) for a in ans]
else:
    print('Impossible')
