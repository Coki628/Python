# -*- coding: utf-8 -*-

"""
・500点自力AC！
・二分探索、最大値の最小化、式変形
・最初、上限を10**18でやったら時間ギリギリで、max(A)*max(B)以上は要らないな、と思って
　10**12にしたら1秒近く速くなった。多分、桁がでかすぎて時間かかっちゃうやつだと思う。
・計算量は20万のNにlogKで、20万*40=800万くらい。pythonTLE、pypyAC1.1秒。
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
sys.setrecursionlimit(10 ** 7)
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

N, K = MAP()
A = LIST()
B = LIST()

# それぞれ昇順と降順で並べれば最適な組にできる
A.sort(reverse=True)
B.sort()

# スコアの最大値をm以内にできるか
def check(m):
    # 全員の合計修行回数k
    k = 0
    for i, a in enumerate(A):
        b = B[i]
        # 式変形： (a-k)*b <= m → a-m/b <= k
        k += max(ceil(a - m/b), 0)
        # スコアをm以内にするために修行回数kがKで足りなくなったらNG
        if k > K:
            return False
    # 修行回数がK以内で全員のスコアをm以内に抑えられればOK
    return True

print(bisearch_min(-1, 10**12, check))
