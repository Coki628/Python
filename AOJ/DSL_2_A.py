# -*- coding: utf-8 -*-

"""
・RMQ(Range Minimum Query)
・セグメント木
・区間minの更新と取得
"""

import sys

sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
INF=2**31-1

class SegTreeMin:
    """
    以下のクエリを処理する
    1.update:  i番目の値をxに更新する
    2.get_min: 区間[l, r)の最小値を得る
    """
 
    def __init__(self, n):
        """
        :param n: 要素数
        """
        self.n = n
        # nより大きい2の冪数
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [INF] * (n2 << 1)
 
    def update(self, i, x):
        """
        i番目の値をxに更新
        :param i: index(0-indexed)
        :param x: update value
        """
        i += self.n2
        self.tree[i] = x
        while i > 1:
            self.tree[i >> 1] = x = min(x, self.tree[i ^ 1])
            i >>= 1
 
    def get_min(self, a, b):
        """
        [a, b)の最小値を得る
        :param a: index(0-indexed)
        :param b: index(0-indexed)
        """
        return self._get_min(a, b, 1, 0, self.n2)
 
    def _get_min(self, a, b, k, l, r):
        """
        [a, b)の最小値を得る内部関数
        :param k: 現在調べている区間のtree内index
        :param l, r: kが表す区間の左右端index [l, r)
        :return: kが表す区間と[a, b)の共通区間内での最小値。共通区間を持たない場合はINF
        """
        # 範囲外ならINF
        if r <= a or b <= l:
            return INF
        # [a,b)が完全に[l,r)を包含するならtree[k]の値を採用
        if a <= l and r <= b:
            return self.tree[k]
        # 一部だけ範囲内なら2つに分けて再帰的に調査
        m = (l + r) // 2
        return min(
            self._get_min(a, b, k << 1, l, m),
            self._get_min(a, b, (k << 1) + 1, m, r)
        )

N,Q=MAP()
st=SegTreeMin(N)
for _ in range(Q):
    com,x,y=MAP()
    if com==0:
        st.update(x, y)
    else:
        print(st.get_min(x, y+1))
