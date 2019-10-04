# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-3、自力AC！
・個数制限なしナップザック系、2次元DP+遷移でもう1ループ系
・計算量約400^3=約6400万は想定通りTLE。
・これから内側の遷移を高速化する。
・セグメント木で内ループを400 => log400に。これでAC、python2.5秒。
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

class SegTree:
    """
    以下のクエリを処理する
    1.update:  i番目の値をxに更新する
    2.query: 区間[l, r)の値を得る
    """
 
    def __init__(self, n, func, init):
        """
        :param n: 要素数(0-indexed)
        :param func: 値の操作に使う関数(min, max, add, gcdなど)
        :param init: 要素の初期値(単位元)
        """
        self.n = n
        self.func = func
        self.init = init
        # nより大きい2の冪数
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [self.init] * (n2 << 1)
 
    def update(self, i, x):
        """
        i番目の値をxに更新
        :param i: index(0-indexed)
        :param x: update value
        """
        i += self.n2
        self.tree[i] = x
        while i > 1:
            self.tree[i >> 1] = x = self.func(x, self.tree[i ^ 1])
            i >>= 1
 
    def query(self, a, b):
        """
        [a, b)の値を得る
        :param a: index(0-indexed)
        :param b: index(0-indexed)
        """
        l = a + self.n2
        r = b + self.n2
        s = self.init
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

N = INT()
phrases = []
for i in range(N):
    l, r, p = MAP()
    phrases.append((l, r, p))
M = INT()
melody = [0] * M
for i in range(M):
    melody[i] = INT()

mxm = max(melody)
dp = [SegTree(mxm+1, max, 0) for i in range(N+1)]
for i in range(1, N+1):
    l, r, p = phrases[i-1]
    for j in range(mxm+1):
        # 上からの遷移
        mx = dp[i-1].query(j, j+1)
        l2, r2 = max(0, j-r), max(0, j-l+1)
        # 取ってこれる範囲があれば、左上と左からの遷移をする
        if not (l2 == r2 == 0):
            mx = max(
                mx,
                dp[i-1].query(l2, r2) + p,
                dp[i].query(l2, r2) + p,
            )
        dp[i].update(j, mx)

ans = [0] * M
for i, mel in enumerate(melody):
    ans[i] = dp[N].query(mel, mel+1)
if min(ans) == 0:
    print(-1)
else:
    [print(a) for a in ans]
