# -*- coding: utf-8 -*-

"""
・自力WAだったものを色々参考で直してTLE。
・そもそもセグ木、普通にminmaxのやつ作ればにぶたんとかいらないじゃん。。
　と分かってこの後根本的に書き直し。
"""

import sys
from operator import itemgetter, add

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
INF = 10 **18
MOD = 10 ** 9 + 7

def compress(A):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(set(A))):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

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
        """ 一点取得 """
        return self.tree[i+self.n2]

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

N = INT()
LR = []
A = set()
for i in range(N):
    l, r = MAP()
    LR.append((l, r))
    A.add(l)
    A.add(r)

zipped, unzipped = compress(A)

LR.sort(key=itemgetter(0))

st1 = SegTree(len(zipped), add, 0)
st2 = SegTree(len(zipped), add, 0)
for i in range(N):
    l, r = LR[i][0], LR[i][1]
    st1.update(zipped[l], st1.get(zipped[l])+1)
    st2.update(zipped[r], st2.get(zipped[r])+1)

def check1(m):
    return st1.query(m, len(zipped)) == 0

def check2(m):
    return st2.query(0, m+1) == 0

def check3(m):
    return st3.query(m, len(zipped)) == 0

def check4(m):
    return st4.query(0, m+1) == 0

ans = 0
# 1区間だけ独立させるケース
for i in range(N):
    l1, r1 = LR[i]
    st1.update(zipped[l1], st1.get(zipped[l1])-1)
    st2.update(zipped[r1], st2.get(zipped[r1])-1)

    idx = bisearch_min(-1, len(zipped), check1) - 1
    l2 = unzipped[idx]
    idx = bisearch_max(0, len(zipped)+1, check2) + 1
    r2 = unzipped[idx]
    ans = max(ans, max(r1-l1+1, 0) + max(r2-l2+1, 0))

    st1.update(zipped[l1], st1.get(zipped[l1])+1)
    st2.update(zipped[r1], st2.get(zipped[r1])+1)

# ある場所から半分に分けるケース
st3 = SegTree(len(zipped), add, 0)
st4 = SegTree(len(zipped), add, 0)
for i in range(N-1):
    l, r = LR[i]
    st1.update(zipped[l], st1.get(zipped[l])-1)
    st2.update(zipped[r], st2.get(zipped[r])-1)
    st3.update(zipped[l], st3.get(zipped[l])+1)
    st4.update(zipped[r], st4.get(zipped[r])+1)

    idx = bisearch_min(-1, len(zipped), check1) - 1
    l1 = unzipped[idx]
    idx = bisearch_max(0, len(zipped)+1, check2) + 1
    r1 = unzipped[idx]
    idx = bisearch_min(-1, len(zipped), check3) - 1
    l2 = unzipped[idx]
    idx = bisearch_max(0, len(zipped)+1, check4) + 1
    r2 = unzipped[idx]
    ans = max(ans, max(r1-l1+1, 0) + max(r2-l2+1, 0))

print(ans)
