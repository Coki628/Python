# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・いもす法、セグメント木で区間min取得
・方針すぐ立ったけど、pythonの手持ちセグ木が遅いから通るか心配だった。
・結果、pythonTLEでpypyACの1.6秒。
・計算量は30万のNへのセグ木更新でNlogN=30万*約18=540万くらい。
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
        return self._get_val(a, b, 1, 0, self.n2)
 
    def _get_val(self, a, b, k, l, r):
        """
        [a, b)の値を得る内部関数
        :param k: 現在調べている区間のtree内index
        :param l, r: kが表す区間の左右端index [l, r)
        :return: kが表す区間と[a, b)の共通区間内での最小値。共通区間を持たない場合は初期値
        """
        # 範囲外なら初期値
        if r <= a or b <= l:
            return self.init
        # [a,b)が完全に[l,r)を包含するならtree[k]の値を採用
        if a <= l and r <= b:
            return self.tree[k]
        # 一部だけ範囲内なら2つに分けて再帰的に調査
        m = (l + r) // 2
        return self.func(
            self._get_val(a, b, k << 1, l, m),
            self._get_val(a, b, (k << 1) + 1, m, r)
        )

N, M = MAP()
ST = []
imos = [0] * (N+2)
for i in range(M):
    s, t = MAP()
    ST.append((s, t))
    imos[s] += 1
    imos[t+1] -= 1
# いもすで区間の重複を集計
imos = list(accumulate(imos))

# 区間min取得のためにセグ木に入れる
st = SegTree(N+2, min, INF)
for i in range(N+2):
    st.update(i, imos[i])

ans = []
for i in range(M):
    # 区間[s,t]に値が1以下になる場所がない(最小値が2以上)ならOK
    s, t = ST[i]
    v = st.get_val(s, t+1)
    if v >= 2:
        ans.append(i+1)

print(len(ans))
for i in ans:
    print(i)
