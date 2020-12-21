# -*- coding: utf-8 -*-

"""
参考：http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3158675#1
・蟻本演習3-3-3
・区間加算BIT(区間加算・区間合計取得)
・10万クエリでpythonAC1.4秒。
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
INF = float('inf')
MOD = 10 ** 9 + 7

class BIT2:
    """ 区間加算BIT(区間加算・区間合計取得) """

    def __init__(self, N):
        self.N = N
        self.data0 = [0] * N
        self.data1 = [0] * N

    def _add(self, data, k, x):
        while k < self.N:
            data[k] += x
            k += k & -k

    def _get(self, data, k):
        s = 0
        while k:
            s += data[k]
            k -= k & -k
        return s

    def add(self, l, r, x):
        """ 区間[l,r)に値xを追加 """

        self._add(self.data0, l, -x*(l-1))
        self._add(self.data0, r, x*(r-1))
        self._add(self.data1, l, x)
        self._add(self.data1, r, -x)

    def query(self, l, r):
        """ 区間[l,r)の和を取得 """

        return self._get(self.data1, r-1) * (r-1) + self._get(self.data0, r-1) \
             - self._get(self.data1, l-1) * (l-1) - self._get(self.data0, l-1)

N, Q = MAP()
bit = BIT2(N+1)

for i in range(Q):
    cmd, *arg = MAP()
    if cmd == 0:
        s, t, x = arg
        bit.add(s, t+1, x)
    else:
        s, t = arg
        print(bit.query(s, t+1))
