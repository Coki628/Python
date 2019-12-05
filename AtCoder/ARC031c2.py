# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc031
・転倒数数え上げ応用、BIT
・案外自分の方針もそう悪くはなかった。
　公式解とはちょっとアプローチの仕方が違うけど、ちょっと直したらACした。
・左右から転倒数(としてない数)を数えておいて、最後に各位置でどっち側がおいしいか比較。
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

ans = 0
for i in range(N):
    # 位置iにある値を左側として使うか右側として使うかの良い方
    ans += min(B1[i], B2[i])
print(ans)
