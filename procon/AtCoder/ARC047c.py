# -*- coding: utf-8 -*-

"""
参考：http://selpog.hateblo.jp/entry/2016/02/24/001000
　　　https://www.slideshare.net/chokudai/arc047-57126901
・階乗進数、BIT上の二分探索
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
        """ 区間[l,r]を右から左に向かってx番目の値がある位置 """

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

    def bisearch_right(self, l, r, x):
        """ 区間[l,r]を左から右に向かってx番目の値がある位置 """

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

class Fact:
    """ 階乗進数 """

    def __init__(self, N):
        """ 最大値をN!として初期化 """

        self.N = N
        self.arr = [0] * N
        for i in range(N):
            self.arr[i] = i

    def div(self, x):
        """ 除算 """

        r = 0
        for i in range(self.N-1, 0, -1):
            tmp = self.arr[i] + r * (i+1)
            self.arr[i] = tmp // x
            r = tmp % x
    
    def _sub(self):
        i = 0
        while self.arr[i] == 0:
            i += 1
        self.arr[i] -= 1
        for j in range(i-1, 0, -1):
            self.arr[j] = j

    def sub(self, x):
        """ 減算 """

        for _ in range(x):
            self._sub()
    
    def to_num(self):
        """ 階乗進数から番号への変換(0-indexed) """

        res = 0
        for i in range(self.N-1, -1, -1):
            res = (i+1) * res + self.arr[i]
        return res

    def to_perm(self):
        """ 階乗進数から順列への変換(0-indexed) """

        bit = BIT(self.N)
        res = [0] * (self.N)
        for i in range(self.N):
            bit.add(i, 1)
        for i in range(self.N-1, -1, -1):
            a = self.arr[i]
            # 残っている数の中でa+1番目のものを使う
            nxt = bit.bisearch_right(0, self.N-1, a+1)
            res[N-i-1] = nxt
            bit.add(nxt, -1)
        return res

N, K = MAP()

fact = Fact(N)
fact.div(K)
ans = fact.to_perm()
[print(a+1) for a in ans]
