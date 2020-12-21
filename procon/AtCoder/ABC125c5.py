# -*- coding: utf-8 -*-

"""
参考：https://tjkendev.github.io/procon-library/python/range_query/rmq_segment_tree.html
・セグメント木
・セグ木改良版、再帰が遅いとの話なので区間取得をwhileでやるようにした。
・pypy1.8秒 => 0.7秒まで劇的に改善。これはやる価値あったな。
"""

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd

def input(): return sys.stdin.readline().strip()
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
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
    2.get_val: 区間[l, r)の値を得る
    """
 
    def __init__(self, n, func, init):
        """
        :param n: 要素数
        :param func: 値の操作に使う関数(min, max, addなど)
        :param init: 要素の初期値
        """
        n += 1
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
 
    def get_val(self, a, b):
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

N=INT()
A=LIST()

st=SegTree(N+2, gcd, 0)
for i in range(N):
    st.update(i+1, A[i])

ans=0
for i in range(1, N+1):
    ans=max(ans, gcd(st.get_val(0, i), st.get_val(i+1, N+2)))
print(ans)
