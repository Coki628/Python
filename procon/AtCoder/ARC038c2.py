"""
参考：https://twitter.com/noshi91/status/1279594849826533377
・蟻本演習4-2-6
・Grundy数、対称戦略、mex取得の高速化、セグ木上の二分探索
・ツイッターで「mex 高速化」を検索すると、どうやらセグ木でできるそうなので、
　「mex セグ木」で再度検索すると、noshiさんがやり方を説明してくれてる。
・ようはセグ木でgrundy数の方をindex、最後に出現したindexを値として持たせる。
　あとは希望する左端位置未満の値が現れる最初のGrundy数がどこかを二分探索すればOK。
　その位置のGrundy数が分かったら、最後に出現したindexを更新する。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
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
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [self.intv] * (n2 << 1)
        if A:
            for i in range(n):
                self.tree[n2+i] = A[i]
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

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()

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
            return -1

    def bisearch_back(self, l, r, x, func):
        """ 区間[l,r]で右から最初にxに対して比較の条件を満たすような値が出現する位置 """

        ok = l - 1
        ng = r + 1
        while ok+1 < ng:
            mid = (ok+ng) // 2
            if func(self.query(mid, r+1), x):
                ok = mid
            else:
                ng = mid
        if ok != l - 1:
            return ok
        else:
            return -1

N = INT()
A = [0] * N
C = [0] * N
for i in range(1, N):
    C[i], A[i] = MAP()

grundy = [0] * N
grundy[0] = 0
# seg[i] := grundy数iが最後に現れたindex
seg = SegTree(N, min, INF, [-1]*N)
seg.update(0, 0)
for i in range(1, N):
    g = seg.bisearch_fore(0, i, i-C[i], lambda a, b: a < b)
    seg.update(g, i)
    # se = set()
    # for j in range(1, C[i]+1):
    #     if i-j >= 0:
    #         se.add(grundy[i-j])
    # g = 0
    # while g in se:
    #     g += 1
    grundy[i] = g

nim = 0
for i in range(1, N):
    nim ^= grundy[i]*(A[i]%2)
if nim:
    print('First')
else:
    print('Second')
