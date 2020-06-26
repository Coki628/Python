"""
・セグ木でもやってみた。こっちのが速い。。pypyAC0.67→0.50秒。
・Sparse Table意味なかったな。。
・あとこどふぉ環境でCEになる文字、「単」だった。他にもあるのかもしれないけど。
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
INF = 10 ** 19
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

    def add(self, i, x):
        self.update(i, self.get(i) + x)

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

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()

for _ in range(INT()):
    N = INT()
    A = LIST()

    B1 = []
    B2 = []
    for i in range(N-1):
        if i % 2 == 0:
            B1.append(A[i+1]-A[i])
        else:
            B2.append(A[i]-A[i+1])
    acc1 = [0] + list(accumulate(B1))
    acc2 = [0] + list(accumulate(B2))
    N1 = len(acc1)
    N2 = len(acc2)
    st1 = SegTree(N1, max, -INF, acc1)
    st2 = SegTree(N2, max, -INF, acc2)

    base = sum([a if i % 2 == 0 else 0 for i, a in enumerate(A)])
    ans = base
    for i in range(N1-1):
        l = acc1[i]
        r = st1.query(i+1, N1)
        ans = max(ans, base+r-l)
    for i in range(N2-1):
        l = acc2[i]
        r = st2.query(i+1, N2)
        ans = max(ans, base+r-l)
    print(ans)
