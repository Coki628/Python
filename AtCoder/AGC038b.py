# -*- coding: utf-8 -*-

"""
参考：https://www.youtube.com/watch?v=RXYksUjlPD4
　　　https://maspypy.com/atcoder-%E5%8F%82%E5%8A%A0%E6%84%9F%E6%83%B3-2019-09-21agc-038
　　　https://img.atcoder.jp/agc038/editorial.pdf
・やっぱり700点の考察は甘くない。出ない。
・セグメント木、2値化して累積和
・普通にセグ木で通ったけど、こういう1つずつ動いていく区間minとかはスライド最小値が使えるそう。
　O(NlogN)がO(N)になる。そのうちやってみるかな。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

class SegTree:
    """
    セグメント木
    1.update:  i番目の値をxに更新する
    2.query: 区間[l, r)の値を得る
    """

    def __init__(self, n, func, intv, A=[]):
        """
        :param n: 要素数(0-indexed)
        :param func: 値の操作に使う関数(min, max, add, gcdなど)
        :param intv: 要素の初期値(単位元)
        :param A: 初期化に使うリスト(オプション)
        """
        self.n = n
        self.func = func
        self.intv = intv
        # nより大きい2の冪数
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [self.intv] * (n2 << 1)
        # 初期化の値が決まっている場合
        if A:
            # 1段目(最下段)の初期化
            for i in range(n):
                self.tree[n2+i] = A[i]
            # 2段目以降の初期化
            for i in range(n2-1, -1, -1):
                self.tree[i] = self.func(self.tree[i*2], self.tree[i*2+1])

    def update(self, i, x):
        """
        i番目の値をxに更新
        :param i: index(0-indexed)
        :param x: update value
        """
        i += self.n2
        self.tree[i] = x
        while i > 0:
            i >>= 1
            self.tree[i] = self.func(self.tree[i*2], self.tree[i*2+1])

    def query(self, a, b):
        """
        [a, b)の値を得る
        :param a: index(0-indexed)
        :param b: index(0-indexed)
        """
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
        """ 一点取得 """
        return self.tree[i+self.n2]

    def all(self):
        """ 全区間[0, n)の取得 """
        return self.tree[1]

N, K = MAP()
A = LIST()

# 隣り合う区間との関係を確認(これで重なる区間同士は全てカバーできる)
stmin = SegTree(N, min, INF, A)
stmax = SegTree(N, max, -INF, A)
total = N - K + 1
dup = 0
for i in range(N-K):
    mn = stmin.query(i, i+K+1)
    mx = stmax.query(i, i+K+1)
    if mn == A[i] and mx == A[i+K]:
        dup += 1

# 離れた区間同士の確認
B = [0] * N
for i in range(N-1):
    # 増加か減少で2値化して累積和
    if A[i] < A[i+1]:
        B[i+1] = 1
B = list(accumulate(B))
# 最初の1回は重複にならないのでINFにしておく
prev = INF
for i in range(N-K+1):
    # 増加がK-1回連続している箇所を確認
    if B[i+K-1] - B[i] == K - 1:
        # 前回の該当箇所と離れている(区間が重ならない)ならカウントする
        if prev + K - 1 < i:
            dup += 1
        prev = i

print(total - dup)
