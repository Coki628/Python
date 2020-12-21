# -*- coding: utf-8 -*-

"""
参考：http://agc005.contest.atcoder.jp/data/agc/005/editorial.pdf
　　　https://maspypy.com/atcoder-参加感想-2019-09-07abc-140
・ABC140eの類題として練習。
・BIT上の二分探索
・試しにBITじゃなくてセグ木使ってみた。→TLE
・最新の非再帰セグ木で再トライ。がしかし変化なくTLE。。
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
    cnt = st.query(m, idx+1)
    return cnt == 0

# idx~mの間に出現済がない(この範囲の中で自分が最小値)かどうか
def calc2(m):
    cnt = st.query(idx, m+1)
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
