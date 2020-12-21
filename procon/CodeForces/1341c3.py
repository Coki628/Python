# -*- coding: utf-8 -*-

"""
・セグ木にぶたんをクラス内に組み込んでみた。
・でも使うのがminかmaxかによって不等号変えなきゃとかあるんだよね。
"""

import sys
from operator import ge

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

class SegTree:

    def __init__(self, n, func, intv, A=[]):
        self.n = n
        self.func = func
        self.intv = intv
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [self.intv] * (n2 << 1)
        if A:
            for i in range(n):
                self.tree[n2+i] = A[i]
            for i in range(n2-1, -1, -1):
                self.tree[i] = self.func(self.tree[i*2], self.tree[i*2+1])

    def update(self, i, x):
        i += self.n2
        self.tree[i] = x
        while i > 0:
            i >>= 1
            self.tree[i] = self.func(self.tree[i*2], self.tree[i*2+1])

    def add(self, i, x):
        self.update(i, self.get(i) + x)

    def query(self, a, b):
        l = a + self.n2
        r = b + self.n2
        s = self.intv
        while l < r:
            if r & 1:
                r -= 1
                s = self.func(s, self.tree[r])
            if l & 1:
                s = self.func(s, self.tree[l])
                l += 1
            l >>= 1
            r >>= 1
        return s

    def get(self, i):
        return self.tree[i+self.n2]

    def all(self):
        return self.tree[1]

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()
    
    def bisect_left(self, l, r, x, op):
        """ 区間[l,r)で最初に値がx以上(以下)になるindexを返す """

        ok = r
        ng = l - 1
        while ng+1 < ok:
            mid = (ok+ng) // 2
            # max,lcmみたいな増える系ならここはx以上(ge)、min,gcdみたいな減る系だとここはx以下(le)になる
            if op(self.query(l, mid+1), x):
                ok = mid
            else:
                ng = mid
        return ok

for _ in range(INT()):
    N = INT()
    P = [(p, i) for i, p in enumerate(LIST())]

    P.sort()
    C = SegTree(N, max, 0, [1]*N)
    for p, idx in P:
        if C.all() != C.get(idx):
            No()
            break
        res = C.bisect_left(idx+1, N, 1, ge)
        if res < N:
            if C.get(res) != 0:
                C.add(res, C.get(idx))
        C.update(idx, 0)
    else:
        Yes()
