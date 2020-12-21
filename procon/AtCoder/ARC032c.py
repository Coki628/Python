# -*- coding: utf-8 -*-

"""
・自力TLE
・試行錯誤したものの、あと1ケースを落とせず。。
・セグ木応用
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

def compress(A):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(set(A))):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

N = INT()
AB = []
S = set()
for i in range(N):
    a, b = MAP()
    AB.append((a, b, i))
    S.add(a)
    S.add(b)

# 座標圧縮
zipped, _ = compress(S)
M = len(zipped)
for i in range(N):
    a, b, i = AB[i]
    AB[i] = (zipped[a], zipped[b], i)

# まず仕事数、同数なら内容の辞書順を比較する
def compare(a, b):
    if a[0] > b[0]:
        return a
    if a[0] < b[0]:
        return b
    for i in range(a[0]):
        if a[1][i] < b[1][i]:
            return a
        elif a[1][i] > b[1][i]:
            return b
    return a

# 独自の比較関数でセグ木を構築
st = SegTree(M, compare, (0, []))
AB.sort()
for a, b, i in AB:
    # 時間aまでで最も良い状況を取ってくる
    cnt, li = st.query(0, a+1)
    # 今回分を合わせて、時間bを更新
    nxt = compare(st.get(b), (cnt + 1, li + [i+1]))
    st.update(b, nxt)
# 全時間帯から、最もうまくいった所を取ってくる
ans = st.all()
print(ans[0])
print(*ans[1])
