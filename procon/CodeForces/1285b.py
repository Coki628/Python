# -*- coding: utf-8 -*-

"""
・なんとか自力AC
・累積和とセグ木を組み合わせて、全区間和中の最大値をNlogNで取得。
・最初、ちょっとセグ木で殴れないかな、と思いつつも、そっちの方針はあまり詰めずに、
　貪欲の考察詰めてWA祭りをしてしまったのがダメだった。
・とはいえ数列の全区間和から特定の値をNlogNで見つけるのは他の問題でも使えそうだし、
　今回やり方確認できて良かった。
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
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

class SegTree:

    def __init__(self, n, func, intv, A=[]):
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
        i += self.n2
        self.tree[i] = x
        while i > 0:
            i >>= 1
            self.tree[i] = self.func(self.tree[i*2], self.tree[i*2+1])

    def query(self, a, b):
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
        return self.tree[i+self.n2]

    def all(self):
        return self.tree[1]

for _ in range(INT()):
    N = INT()
    A = LIST()

    x = sum(A)
    acc = [0] + list(accumulate(A))
    st = SegTree(N+1, max, -INF, acc)
    y = st.query(1, N)
    for i in range(1, N):
        y = max(y, st.query(i+1, N+1) - st.get(i))
    if x > y:
        YES()
    else:
        NO()
