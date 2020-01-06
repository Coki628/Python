# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・速攻自力AC
・二分探索
・これ懐かしい！この回、Cが簡単だったとかでかなり昔なのにDに挑戦した時だ。
　結果、完敗で二分探索訳分からんになってたのは覚えてる。
　今だったら速攻だけど、この時はこれなんて解ける気がしなかったんだよな。
　まあ、さすがに1年半近く前だし当然なんだけど、成長を感じて感慨深い。
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

def check(m):
    # 全体攻撃
    b = B * m
    H2 = [h-b for h in H]
    # 単体攻撃
    cnt = 0
    for h in H2:
        if h > 0:
            cnt += ceil(h, A)
    # 必要な単体攻撃の回数がm以下ならOK
    return cnt <= m

N, A, B = MAP()
H = LIST(N)
# Aを単体攻撃、Bを全体攻撃とする
A -= B

# 二分探索
res = bisearch_min(0, 10**9, check)
print(res)
