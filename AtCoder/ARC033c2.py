# -*- coding: utf-8 -*-

"""
・蟻本演習3-3-2、自力AC
・BIT上の二分探索
・計算量は20万にlog2つ、20万*300ちょっとで6000万。
　まあ更新の回はlog1つだからこれより全然少ないはずだけど。
・pythonTLE,pypyAC0.8秒。
・BITにぶたんをBITクラスに組み込んだ。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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
        """ 区間l,rで左からx番目の値がある位置 """
        l_val = self.sum(l)
        ok = r
        ng = l
        while ng+1 < ok:
            mid = (ok+ng) // 2
            if self.sum(mid) - l_val >= x:
                ok = mid
            else:
                ng = mid
        return ok

    def bisearch_right(self, l, r, x):
        """ 区間l,rで右からx番目の値がある位置 """
        r_val = self.sum(r)
        ok = r
        ng = l
        while ng+1 < ok:
            mid = (ok+ng) // 2
            if r_val - self.sum(mid) < x:
                ok = mid
            else:
                ng = mid
        return ok

N = 2 * 10 ** 5
bit = BIT(N+1)
for i in range(INT()):
    t, x = MAP()
    if t == 1:
        bit.add(x, 1)
    else:
        res = bit.bisearch_left(0, N, x)
        print(res)
        bit.add(res, -1)
