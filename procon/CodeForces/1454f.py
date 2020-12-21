"""
・セグ木にぶたん。pypyTLE。こどふぉはpypyに慈悲なし。C++でAC。コメントとかもそっちで。
"""

import sys
from itertools import accumulate
from operator import ge, gt, le, lt

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

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

    def bisearch_fore(self, l, r, x, func):

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

for _ in range(INT()):
    N = INT()
    A = LIST()

    segmx = SegTree(N, max, -INF, A)
    segmn = SegTree(N, min, INF, A)
    accmx = list(accumulate(A, max))

    for i in range(1, N-1):
        x = accmx[i-1]
        l1 = segmn.bisearch_fore(i, N-1, x, le)
        r1 = segmn.bisearch_fore(i, N-1, x, lt)
        if r1 == -1:
            r1 = N - 1

        l2 = segmx.bisearch_back(i, N-1, x, gt)
        r2 = segmx.bisearch_back(i, N-1, x, ge)
        if l2 == -1:
            l2 = 0
        if r2 == -1:
            r2 = 0

        if r1 <= l2 or l1 >= r2:
            continue
        if l1 == r1 and A[l1] != x:
            continue
        if l2 == r2 and A[l2] != x:
            continue
        # print(i, x, l1, r1, l2, r2)
        # print(i, min(r1, r2))
        j = min(r1, r2)
        YES()
        print(i, j-i, N-j)
        break
    else:
        NO()
