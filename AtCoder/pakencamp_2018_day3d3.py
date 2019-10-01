# -*- coding: utf-8 -*-

"""
参考：https://tjkendev.github.io/procon-library/python/range_query/rmq_segment_tree.html
・累積和とセグメント木
・セグ木改良版、再帰が遅いとの話なので区間取得をwhileでやるようにした。
・pypy1.5秒 => 0.8秒まで劇的に改善。これはやる価値あったな。
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

N = INT()
A = [0] + LIST(N-1) + [0]

# 区間和取得のために累積和
acc = list(accumulate(A))
rev_acc = list(accumulate(A[::-1]))[::-1]

# 区間和から最大を取るためのセグ木
st = SegTree(N+1, max, -INF)
rev_st = SegTree(N+1, max, -INF)
for i, a in enumerate(acc):
    st.update(i, a)
for i, a in enumerate(rev_acc):
    rev_st.update(i, a)

for i in range(N):
    # 動かない、右に出る、左に出る、から最大を取る
    ans = max(
        0,
        st.get_val(i+1, N+1) - acc[i],
        rev_st.get_val(0, i+1) - rev_acc[i+1]
    )
    print(ans)
