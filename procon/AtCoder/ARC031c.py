# -*- coding: utf-8 -*-

"""
・自力WA
・転倒数数え上げっぽいしBIT使いそう、の判断は良かった。
・多分、これだと境界を横断して移動させた方が良いやつを考慮できてなかったんだと思う。
"""

import sys
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
MOD = 10 ** 9 + 7

class BIT:
    """ Binary Indexed Tree """

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

    def bisearch_left(self, l, r, x):
        """ 区間[l,r]で左からx番目の値がある位置 """
        l_sm = self.sum(l-1)
        ok = r + 1
        ng = l - 1
        while ng+1 < ok:
            mid = (ok+ng) // 2
            if self.sum(mid) - l_sm >= x:
                ok = mid
            else:
                ng = mid
        if ok != r + 1:
            return ok
        else:
            return -1

    def bisearch_right(self, l, r, x):
        """ 区間[l,r]で右からx番目の値がある位置 """
        r_sm = self.sum(r)
        ok = l - 1
        ng = r + 1
        while ok+1 < ng:
            mid = (ok+ng) // 2
            if r_sm - self.sum(mid-1) >= x:
                ok = mid
            else:
                ng = mid
        if ok != l - 1:
            return ok
        else:
            return -1

N = INT()
A = LIST()

bit1 = BIT(N+1)
B1 = [0] * N
for i in range(N):
    a = A[i]
    # 自分より左にある転倒数
    B1[i] = i - bit1.sum(a)
    bit1.add(a, 1)

bit2 = BIT(N+1)
B2 = [0] * N
for i in range(N-1, -1, -1):
    a = A[i]
    # 自分より右にある転倒してない数
    B2[i] = (N-i-1) - bit2.sum(a)
    bit2.add(a, 1) 

# 左から累積和
B1 = list(accumulate(B1))
# 右から累積和
B2 = list(accumulate(B2[::-1]))[::-1]

ans = INF
for i in range(1, N):
    # i-1まで増加させてiから減少、を全箇所見る
    ans = min(ans, B1[i-1] + B2[i])
print(ans)
