# -*- coding: utf-8 -*-

"""
参考：https://rsk0315.hatenablog.com/entry/2020/05/02/181650
・区間加算・1点取得が普通のセグ木でもできるという知見を頂いたのでやってみる。
・確かにこうやっていもすっぽくやるとできるんだな、頭いい。
・速くなるかと思ったけど、BITは区間加算のやつも普通のも大差ないのかな。pypyAC1.2→1.1秒。
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
        n += 1
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

    def update(self, i, x):
        """ 値の更新：添字i, 値x """
        self.add(i, x - self.get(i))

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
bit = BIT(H)
ans = [0] * Q
for event in events:
    if event[1] == 0:
        _, _, y1, y2, c = event
        # 区間加算
        bit.add(y1, c)
        bit.add(y2+1, -c)
    elif event[1] == 1:
        _, _, y, idx = event
        # 1点取得
        ans[idx] = bit.sum(y)
    else:
        _, _, y1, y2, c = event
        # 区間加算
        bit.add(y1, -c)
        bit.add(y2+1, c)
[print(a) for a in ans]
