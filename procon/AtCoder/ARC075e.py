# -*- coding: utf-8 -*-

"""
参考：https://www.youtube.com/watch?v=XTW-Q5buWOk
　　　https://img.atcoder.jp/arc075/editorial.pdf
・蟻本演習3-1-4
・平均を総和にするテク、累積和、BIT、転倒数
・解説はPDFよりは動画のが分かると思う。
・ここでは上記テクを累積和・BITと組み合わせて、高速に平均がK以上になる区間のペア数を出す。
・計算量は20万のNにlogだから400万くらい。pypyAC0.5秒、意外とpythonも間に合ってAC1.6秒。
"""

import sys
from itertools import accumulate

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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

class BIT:

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

def compress(A):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(set(A))):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

N, K = MAP()
# 予め-Kした状態で用意しておく
A = [a-K for a in LIST(N)]

# 累積和
acc = [0] + list(accumulate(A))
# 座標圧縮
zipped, _ = compress(acc)

bit = BIT(len(zipped))
cnt = 0
for a in acc:
    # 自分より左にある、自分の値以下の数(転倒数の数え上げ)
    cnt += bit.sum(zipped[a])
    bit.add(zipped[a], 1)
print(cnt)
