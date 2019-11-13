# -*- coding: utf-8 -*-

"""
参考：https://smijake3.hatenablog.com/entry/2018/11/03/100133
・蟻本演習3-3-3
・遅延評価セグメント木(区間更新・区間合計取得)
・10万クエリでpythonAC4.5秒。
"""

import sys
from operator import add

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
INF = 2 ** 31 - 1
MOD = 10 ** 9 + 7

class SegTreeLazy:
    """ 遅延評価セグメント木(区間更新・区間合計取得) """

    def __init__(self, N, func, intv):
        self.intv = intv
        self.func = func
        self.LV = (N-1).bit_length()
        self.N0 = 2**self.LV
        self.data = [intv]*(2*self.N0)
        self.lazy = [None]*(2*self.N0)

    # 伝搬される区間のインデックス(1-indexed)を全て列挙するgenerator
    def gindex(self, l, r):
        L = (l + self.N0) >> 1; R = (r + self.N0) >> 1
        lc = 0 if l & 1 else (L & -L).bit_length()
        rc = 0 if r & 1 else (R & -R).bit_length()
        v = 2
        for i in range(self.LV):
            if rc <= i:
                yield R
            if L < R and lc <= i:
                yield L
            L >>= 1; R >>= 1; v <<= 1

    # 遅延評価の伝搬処理
    def propagates(self, *ids):
        # 1-indexedで単調増加のインデックスリスト
        for i in reversed(ids):
            v = self.lazy[i-1]
            if v is None:
                continue
            self.lazy[2*i-1] = self.data[2*i-1] = self.lazy[2*i] = self.data[2*i] = v >> 1
            self.lazy[i-1] = None

    def update(self, l, r, x):
        """ 区間[l,r)の値をxに更新 """

        *ids, = self.gindex(l, r)
        # 1. トップダウンにlazyの値を伝搬
        self.propagates(*ids)
        # 2. 区間[l,r)のdata, lazyの値を更新
        L = self.N0 + l; R = self.N0 + r
        v = x
        while L < R:
            if R & 1:
                R -= 1
                self.lazy[R-1] = self.data[R-1] = v
            if L & 1:
                self.lazy[L-1] = self.data[L-1] = v
                L += 1
            L >>= 1; R >>= 1; v <<= 1
        # 3. 伝搬させた区間について、ボトムアップにdataの値を伝搬する
        for i in ids:
            self.data[i-1] = self.func(self.data[2*i-1], self.data[2*i])

    def query(self, l, r):
        """ 区間[l,r)の和を取得 """

        # 1. トップダウンにlazyの値を伝搬
        self.propagates(*self.gindex(l, r))
        L = self.N0 + l; R = self.N0 + r

        # 2. 区間[l, r)の和を求める
        s = self.intv
        while L < R:
            if R & 1:
                R -= 1
                s = self.func(s, self.data[R-1])
            if L & 1:
                s = self.func(s, self.data[L-1])
                L += 1
            L >>= 1; R >>= 1
        return s

N, Q = MAP()
stl = SegTreeLazy(N+1, add, 0)

ans = []
for i in range(Q):
    cmd, *arg = MAP()
    if cmd == 0:
        s, t, x = arg
        stl.update(s, t+1, x)
    else:
        s, t = arg
        ans.append(str(stl.query(s, t+1)))
print('\n'.join(ans))
