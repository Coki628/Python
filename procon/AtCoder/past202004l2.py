# -*- coding: utf-8 -*-

"""
・indexセグ木修正版。
・無事AC。log1つになって、速度もpypyで0.6→0.2秒まで速くなった。
　TLEだったpythonもなんとかAC1.8秒。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

class SegTreeIndex:
    """
    セグメント木
    1.update:  i番目の値をxに更新する
    2.query: 区間[l, r)の値とindex(同値があった場合は一番左)を得る
    """

    def __init__(self, n, func, intv, A=[]):
        """
        :param n: 要素数(0-indexed)
        :param func: 値の操作に使う関数(min, maxなど)
        :param intv: 要素の初期値(単位元)
        :param A: 初期化に使うリスト(オプション)
        """
        self.n = n
        self.func = func
        self.intv = (intv, -1)
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [self.intv] * (n2 << 1)
        if A:
            for i in range(n):
                self.tree[n2+i] = (A[i], i)
            for i in range(n2-1, -1, -1):
                self.tree[i] = self.compare(self.tree[i*2], self.tree[i*2+1])

    def compare(self, a, b):
        if a[0] == b[0]:
            # 同値はindexが小さい方優先
            if a[1] <= b[1]:
                return a
            else:
                return b
        elif self.func(a[0], b[0]) == a[0]:
            return a
        else:
            return b

    def update(self, i, x):
        """
        i番目の値をxに更新
        :param i: index(0-indexed)
        :param x: update value
        """
        i += self.n2
        self.tree[i] = (x, i-self.n2)
        while i > 0:
            i >>= 1
            self.tree[i] = self.compare(self.tree[i*2], self.tree[i*2+1])

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
                s = self.compare(s, self.tree[r])
            if l & 1:
                s = self.compare(s, self.tree[l])
                l += 1
            l >>= 1
            r >>= 1
        return s

    def get(self, i):
        """ 一点取得 """
        return self.tree[i+self.n2][0]

    def all(self):
        """ 全区間[0, n)の取得 """
        return self.tree[1]

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()

N, K, D = MAP()
A = [0] + LIST()

if D * (K-1) + 1 > N:
    print(-1)
    exit()

sti = SegTreeIndex(N+1, min, INF, A)
ans = []
cur = 0
for i in range(K):
    l = cur + (D if i != 0 else 1)
    r = N + 1 - (D * (K-i-1) + 1)
    mn, cur = sti.query(l, r+1)
    ans.append(A[cur])
print(*ans)
