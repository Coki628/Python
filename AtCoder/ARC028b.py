# -*- coding: utf-8 -*-

"""
・自力AC！
・BIT上の二分探索
・ABC140eでやってたのが役に立った。
・計算量は10万のNに二分探索とBITの区間取得でO(NlogNlogN)=10万*300弱で3000万。
・pythonさすがにTLEでpypyAC0.5秒。
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

class BIT:

    def __init__(self, n):
        # 0-indexed
        nv = 1
        while nv < n:
            nv *= 2
        self.size = nv
        self.tree = [0] * nv

    def sum(self, i):
        """ [0, i]を合計する """
        s = 0
        i += 1
        while i > 0:
            s += self.tree[i-1]
            i -= i & -i
        return s

    def add(self, i, x):
        """ 値の追加：添字i, 値x """
        i += 1
        while i <= self.size:
            self.tree[i-1] += x
            i += i & -i

    def get(self, l, r=None):
        """ 区間和の取得 [l, r) """
        # 引数が1つなら一点の値を取得
        if r is None: r = l + 1
        res = 0
        if r: res += self.sum(r-1)
        if l: res -= self.sum(l-1)
        return res

N, K = MAP()
# i:順位, x:年齢
itox = [x-1 for x in LIST()]
xtoi = [0] * N
for i, x in enumerate(itox):
    xtoi[x] = i

# K-1までは予め入れておく
bit = BIT(N)
for i in range(K-1):
    bit.add(itox[i], 1)

# 区間和がK(人)以上になるかどうか
def calc(m):
    cnt = bit.sum(m)
    return cnt >= K

for i in range(K-1, N):
    bit.add(itox[i], 1)
    # i位までの中で、年齢K番目の人の位置
    x = bisearch_min(-1, N, calc)
    print(xtoi[x]+1)
