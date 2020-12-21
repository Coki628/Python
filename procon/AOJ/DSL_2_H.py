# -*- coding: utf-8 -*-

"""
参考：https://smijake3.hatenablog.com/entry/2018/11/03/100133
・蟻本演習3-3-3
・Starry Sky Tree(区間加算・区間最小(大)値取得)
・10万クエリでpythonAC4.3秒。
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
INF = 2 ** 31 - 1
MOD = 10 ** 9 + 7

class StarrySkyTree:
    """ Starry Sky Tree(区間加算・区間最小(大)値取得) """

    def __init__(self, N, func, intv):
        self.intv = intv
        self.func = func
        LV = (N-1).bit_length()
        self.N0 = 2**LV
        self.data = [0]*(2*self.N0)
        self.lazy = [0]*(2*self.N0)

    # 伝搬される区間のインデックス(1-indexed)を全て列挙するgenerator
    def gindex(self, l, r):
        L = l + self.N0; R = r + self.N0
        lm = (L // (L & -L)) >> 1
        rm = (R // (R & -R)) >> 1
        while L < R:
            if R <= rm:
                yield R
            if L <= lm:
                yield L
            L >>= 1; R >>= 1
        while L:
            yield L
            L >>= 1

    # 遅延評価の伝搬処理
    def propagates(self, *ids):
        # 1-indexedで単調増加のインデックスリスト
        for i in reversed(ids):
            v = self.lazy[i-1]
            if not v:
                continue
            self.lazy[2*i-1] += v; self.lazy[2*i] += v
            self.data[2*i-1] += v; self.data[2*i] += v
            self.lazy[i-1] = 0

    def update(self, l, r, x):
        """ 区間[l,r)の値にxを加算 """

        # 1. lazyの値は伝搬させない
        # 2. 区間[l,r)のdata, lazyの値を更新
        L = self.N0 + l; R = self.N0 + r
        while L < R:
            if R & 1:
                R -= 1
                self.lazy[R-1] += x; self.data[R-1] += x
            if L & 1:
                self.lazy[L-1] += x; self.data[L-1] += x
                L += 1
            L >>= 1; R >>= 1
        # 3. 更新される区間を部分的に含んだ区間のdataの値を更新 (lazyの値を考慮)
        for i in self.gindex(l, r):
            self.data[i-1] = self.func(self.data[2*i-1], self.data[2*i]) + self.lazy[i-1]

    def query(self, l, r):
        """ 区間[l,r)の最小(大)値を取得 """

        # 1. トップダウンにlazyの値を伝搬
        self.propagates(*self.gindex(l, r))
        L = self.N0 + l; R = self.N0 + r

        # 2. 区間[l, r)の最小(大)値を求める
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
sst = StarrySkyTree(N+1, min, INF)

ans = []
for i in range(Q):
    cmd, *arg = MAP()
    if cmd == 0:
        s, t, x = arg
        sst.update(s, t+1, x)
    else:
        s, t = arg
        ans.append(str(sst.query(s, t+1)))
print('\n'.join(ans))
