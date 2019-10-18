# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/70620
・セグメント木、二分探索
・2周でなく3周する。
・後ろから決めていく。
・セグ木上の二分探索で条件を満たす位置を探せる。
・10万にlog2つ乗るのつらくて、pypyTLE。。
"""

import sys

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
    2.query: 区間[l, r)の値を得る
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
 
    def query(self, a, b):
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
A = LIST()

ans = [-1] * N
if min(A) >= max(A) / 2:
    print(*ans)
    exit()

st1 = SegTree(N*3, max, -INF)
st2 = SegTree(N*3, min, INF)
for i in range(N):
    st1.update(i, A[i])
    st1.update(i+N, A[i])
    st1.update(i+N*2, A[i])
    st2.update(i, A[i])
    st2.update(i+N, A[i])
    st2.update(i+N*2, A[i])

# 区間内にA[i]より大きい値があるかどうか
def calc1(m):
    res = st1.query(i, m+1)
    return res > A[i%N]

# 区間内にA[i]/2より小さい値があるかどうか
def calc2(m):
    res = st2.query(i, m+1)
    return res < A[i%N] / 2

dp = [INF] * (N*3)
for i in range(N*3-1, -1, -1):
    # 自分より先で、条件を満たす値が最初に出現する位置
    j = bisearch_min(i, N*3, calc1)
    k = bisearch_min(i, N*3, calc2)
    if j < k:
        # 自分より大きい値が先に来るなら、そこを参照する
        dp[i] = dp[j] + (j - i)
    elif j > k:
        # 自分/2より小さい値が先に来るなら、自分が決める
        dp[i] = k - i

print(*dp[:N])
