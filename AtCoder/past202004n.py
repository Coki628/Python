# -*- coding: utf-8 -*-

"""
・なんとか時間内自力AC！
・座圧、平面走査、区間加算BIT
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

    def get(self, i):
        return self.query(i, i+1)
    
    def update(self, i, x):
        self.add(i, i+1, x - self.get(i))

def compress(S):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(S)):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

N, Q = MAP()
S1 = set()
S2 = set()
XY = []
for _ in range(N):
    x1, y1, d, c = MAP()
    x2 = x1 + d
    y2 = y1 + d
    XY.append((x1, y1, x2, y2, c))
    S1.add(x1)
    S1.add(x2)
    S2.add(y1)
    S2.add(y2)
Qs = []
for _ in range(Q):
    x, y = MAP()
    Qs.append((x, y))
    S1.add(x)
    S2.add(y)

zpx, unzpx = compress(S1)
zpy, unzpy = compress(S2)
W, H = len(zpx), len(zpy)
events = []
for i in range(N):
    x1, y1, x2, y2, c = XY[i]
    events.append((zpx[x1], 0, zpy[y1], zpy[y2], c))
    events.append((zpx[x2], 2, zpy[y1], zpy[y2], c))
for i in range(Q):
    x, y = Qs[i]
    events.append((zpx[x], 1, zpy[y], i))

events.sort()
bit = BIT2(H)
ans = [0] * Q
for event in events:
    if event[1] == 0:
        _, _, y1, y2, c = event
        bit.add(y1, y2+1, c)
    elif event[1] == 1:
        _, _, y, idx = event
        ans[idx] = bit.get(y)
    else:
        _, _, y1, y2, c = event
        bit.add(y1, y2+1, -c)
[print(a) for a in ans]
