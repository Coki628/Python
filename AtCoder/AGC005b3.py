# -*- coding: utf-8 -*-

"""
参考：http://agc005.contest.atcoder.jp/data/agc/005/editorial.pdf
　　　https://maspypy.com/atcoder-参加感想-2019-09-07abc-140
・ABC140eの類題として練習。
・BIT上の二分探索
・試しにBITじゃなくてセグ木使ってみた。→TLE
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

def bisearch_min(mn, mx, func):
    """ 条件を満たす最小値を見つける二分探索 """
    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            # 下を探しに行く
            ok = mid
        else:
            # 上を探しに行く
            ng = mid
    return ok

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索 """
    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

# m~idxの間に出現済がない(この範囲の中で自分が最小値)かどうか
def calc1(m):
    cnt = st.get_val(m, idx+1)
    return cnt == 0

# idx~mの間に出現済がない(この範囲の中で自分が最小値)かどうか
def calc2(m):
    cnt = st.get_val(idx, m+1)
    return cnt == 0

N = INT()
A = LIST()
# aの昇順に処理できるようにindexで並べておく
idxs = [0] * (N+1)
for i, a in enumerate(A):
    idxs[a] = i + 1

st = SegTree(N+2, add, 0)
# 先頭と末尾に番兵を仕込む
st.update(0, 1)
st.update(N+1, 1)
ans = [0] * (N+1)
for a in range(1, N+1):
    # a(1~N)が格納されているindex
    idx = idxs[a]
    # 自分より小さいindexで最初に自分より小さい値がある直前の場所
    l = bisearch_min(0, idx+1, calc1)
    # 自分より大きいindexで最初に自分より小さい値がある直前の場所
    r = bisearch_max(idx, N+1, calc2)
    # aを使う回数 * a = 左端として使える範囲 * 右端として使える範囲 * a
    ans[a] = (idx-l+1) * (r-idx+1) * a
    # aを出現済とする
    st.update(idx, 1)
# 全てのaについての合計
print(sum(ans))
