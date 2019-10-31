# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc037
・蟻本演習3-1-1
・二分探索、式変形
・にぶたん内でにぶたん系
・小さい方からK番目がX => X未満の数がK個未満となる最大値を探す。
・こういう言い換えはよくあるけど、自分で導くのほんと難しい。
・判定部分では式変形してここでも二分探索。
・計算量は3万にlog2つかかる感じ。30000*15*60=2700万くらい。
・きついと思ったけどpythonでも通って、pythonAC1.3秒、pypyAC1.0秒。
・解説通りに、以下の言い方にした版。
・小さい方からK番目がX => X以下の数がK個以上となる最小値を探す。
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

# x以下の数がK個以上かどうか
def calc(x):
    # 各aについて、条件を満たす数を数える
    ans = 0
    for a in A:
        # 式変形： a * b <= x => b <= floor(x / a)
        ans += bisect_right(B, x // a)
    return ans >= K

N, K = MAP()
A = sorted(LIST())
B = sorted(LIST())
MAX = 10 ** 18

print(bisearch_min(0, MAX+1, calc))
