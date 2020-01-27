# -*- coding: utf-8 -*-

"""
・600点自力AC！今回はこれに救われた。。
・区間加算・1点取得ができれば、あとは左から貪欲。
・StarrySkyTreeで殴った。
・20万のNでSSTのlog乗って、間に合うか心配だったけど、pypyAC1.7秒。
・SSTの代わりにBIT2を使った版。こっちのがちょっと速くてpypyAC1.4秒。
・ソートにitemgetter使ったら速くなって1.1秒。
"""

import sys
from bisect import bisect_right
from operator import itemgetter

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

class BIT2:
    """ 区間加算BIT(区間加算・区間合計取得) """

    def __init__(self, N):
        # 添字0が使えないので、内部的には全て1-indexedとして扱う
        N += 1
        self.N = N
        self.data0 = [0] * N
        self.data1 = [0] * N

    def _add(self, data, k, x):
        k += 1
        while k < self.N:
            data[k] += x
            k += k & -k

    def _get(self, data, k):
        k += 1
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

N, D, K = MAP()

XH = []
for i in range(N):
    x, h = MAP()
    XH.append((x, h))

XH.sort(key=itemgetter(0))
X, _ = zip(*XH)
D = D * 2

bit = BIT2(N)
ans = 0
for i, (x, h) in enumerate(XH):
    # 前回までの爆撃でのダメージを引く
    h -= bit.query(i, i+1) * K
    if h <= 0:
        continue
    # 今回必要な爆撃回数
    cnt = ceil(h, K)
    # 今回の爆撃が届く右端
    idx = bisect_right(X, x+D) - 1
    # 今回加算される爆撃回数を区間加算
    bit.add(i, idx+1, cnt)
    ans += cnt
print(ans)
