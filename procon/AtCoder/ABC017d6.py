# -*- coding: utf-8 -*-

"""
・蟻本演習3-2-2、自力AC！
・尺取法、貰うDP、セグ木で区間和取得
・尺取とDPを組み合わせて使う。
"""

import sys
from operator import add

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
        """ 1点取得 """
        return self.tree[i+self.n2]

N, M = MAP()
A = LIST(N)

# 貰うDPで区間取得したいのでセグ木でDPを構築
dp = SegTree(N+1, add, 0)
dp.update(0, 1)

# 尺取法
l = r = 0
S = set()
while l < N:
    # 未使用サプリが続く限り、区間を伸ばす
    while r < N and A[r] not in S:
        S.add(A[r])
        r += 1
        # 有効な区間から貰うDPする
        sm = dp.query(l, r)
        sm %= MOD
        dp.update(r, sm)
    if l == r:
        # 左が右に追いついたら、右も左に合わせて+1する
        r += 1
    else:
        # 左を動かす分、集合から引く
        S.remove(A[l])
    l += 1
print(dp.get(N))
