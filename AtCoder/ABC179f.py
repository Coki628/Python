"""
・終了後1分8秒で自力AC、、、悔しい。。。
・セグ木上の二分探索
・seg[i] := i列目は何行目まで白くなったか
　みたいなやつを更新しながら、i列目を白くする時には「i列目以降まで白くなってる最初の行」を
　二分探索で出して、その手前までは色変できるね、ってやった。
・公式解はもっと頭良く線形で解いてたけど。。。
"""

import sys
from operator import ge

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class SegTree:
    """
    セグメント木
    1.update: i番目の値をxに更新する
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

    def add(self, i, x):
        self.update(i, self.get(i) + x)

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

    def bisearch_fore(self, l, r, x, func):
        """ 区間[l,r]で左から最初にxに対して比較の条件を満たすような値が出現する位置 """

        ok = r + 1
        ng = l - 1
        while ng+1 < ok:
            mid = (ok+ng) // 2
            if func(self.query(l, mid+1), x):
                ok = mid
            else:
                ng = mid
        if ok != r + 1:
            return ok
        else:
            return INF

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()

N, Q = MAP()
ans = (N-2) * (N-2)
seg1 = SegTree(N+1, max, -INF, [0]*(N+1))
seg2 = SegTree(N+1, max, -INF, [0]*(N+1))
seg1.update(N, N)
seg2.update(N, N)
for _ in range(Q):
    t, x = MAP()
    if t == 1:
        res = seg1.bisearch_fore(0, N, x, ge)
        ans -= res - 2
        seg2.update(x, res)
    else:
        res = seg2.bisearch_fore(0, N, x, ge)
        ans -= res - 2
        seg1.update(x, res)
print(ans)
