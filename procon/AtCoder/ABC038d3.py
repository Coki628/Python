# -*- coding: utf-8 -*-

"""
参考：http://abc038.contest.atcoder.jp/data/abc/038/editorial.pdf
・セグメント木でやった版
・BITでやるより遅かった。pypyでギリギリのAC
"""

import sys
from operator import itemgetter

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
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

N=INT()
HW=[]
for i in range(N):
    w,h=MAP()
    # 紛らわしいので縦横の順にしておく
    HW.append((h, w))

# 縦の昇順、横の降順に並べておく(横に降順の理由は公式解説と自分のノート参照)
HW.sort(key=itemgetter(1), reverse=True)
HW.sort(key=itemgetter(0))

# dp[i] := i番目の箱を最も外側とするとき、最大で何重の入れ子とできるか
dp=[0]*N
# st[i] := 最も外側の箱の横の長さがiとなる入れ子の数の最大値
st=SegTree(100001, max, 0)

# 縦が小さい方から決めていく
for i in range(N):
    h,w=HW[i]
    # 横が自分より小さいものの中から最大を取る(+自分の分で1)
    dp[i]=st.get_val(0, w)+1
    # 今回分は決まったので、それに合わせてSegTreeも更新
    st.update(w, dp[i])

print(max(dp))
