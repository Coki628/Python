# -*- coding: utf-8 -*-

"""
・RSQ(Range Sum Query)
・BIT(Binary Indexed Tree)
・区間和の更新と取得
"""

import sys

sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
INF=float('inf')

class BIT:
    # N+1で初期化すること
    def __init__(self, n):
        # 0-indexed
        nv = 1
        while nv < n:
            nv *= 2
        self.size = nv
        self.tree = [0] * nv

    # [0, i]を合計する
    def sum(self, i):
        s = 0
        i += 1
        while i > 0:
            s += self.tree[i-1]
            i -= i & -i
        return s

    # 値の追加：添字, 値
    def add(self, i, x):
        i += 1
        while i <= self.size:
            self.tree[i-1] += x
            i += i & -i

    # 区間和の取得 [l, r)
    def get(self, l, r=None):
        # 引数が1つなら一点の値を取得
        if r is None: r = l+1
        res = 0
        if r: res += self.sum(r-1)
        if l: res -= self.sum(l-1)
        return res

N,Q=MAP()
bit=BIT(N+1)
for _ in range(Q):
    com,x,y=MAP()
    if com==0:
        bit.add(x, y)
    else:
        print(bit.get(x, y+1))
