# -*- coding: utf-8 -*-

"""
参考：http://mayokoex.hatenablog.com/entry/2016/03/19/230238
・小数の二分探索
・時間tですべての頂点が一箇所に集まることが出来るか
・集まる座標位置を探すのではなく、時間で可否を見る視点
・見方が分かったら、すぐ自力実装はできた。
・ちゃんと考察して自分で見えるようになりたいなぁ。。
・二分探索ライブラリ確認用
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
X = Y = 0
XYC = []
for i in range(N):
    x, y, c = MAP()
    XYC.append((x, y, c))
    X += x
    Y += y

# 時間tで全部の点が集まれる範囲が存在するかどうか
def calc(t):
    L = D = -INF
    R = U = INF
    for x, y, c in XYC:
        # この点が時間tで行ける上下左右の範囲
        l = x - t/c
        r = x + t/c
        d = y - t/c
        u = y + t/c
        # 最大最小を取る
        L = max(L, l)
        R = min(R, r)
        D = max(D, d)
        U = min(U, u)
    # 全部の点が行ける範囲が残っていればOK
    return L <= R and D <= U

def bisearch_min(mn, mx, func):
    """ 条件を満たす最小値を見つける二分探索(小数用) """
    ok = mx
    ng = mn
    for i in range(100):
        mid = (ok+ng) / 2
        if func(mid):
            # 下を探しに行く
            ok = mid
        else:
            # 上を探しに行く
            ng = mid
    return ok

# 小数の二分探索
print(bisearch_min(-1, 2*10**9, calc))
