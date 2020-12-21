# -*- coding: utf-8 -*-

"""
・貰うDPの遷移をBITで高速化。
・計算量は50万のNにlog、pythonTLE,pypyMLE。。制約64MBつれー。
"""

import sys
from bisect import bisect_left
from itertools import accumulate

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
MOD = 1234567

class BIT:
    """ Binary Indexed Tree """

    def __init__(self, n):
        # 0-indexed
        n += 1
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

    def update(self, i, x):
        """ 値の更新：添字i, 値x """
        self.add(i, x - self.get(i))

N, K = MAP()
A = [0] + list(accumulate(LIST(N)))

bit = BIT(N+1)
bit.add(0, 1)
for i, a in enumerate(A[1:], 1):
    # どこまで前から跳んでこれるか
    prev = bisect_left(A, max(a-K, 0))
    # 届く範囲全てから通り数を集めてくる
    sm = bit.get(prev, i) % MOD
    bit.add(i, sm)
ans = bit.get(N)
print(ans)
