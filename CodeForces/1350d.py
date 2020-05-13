"""
・自力WA
・みんな落としてた。たとえ時間があっても正しい考察を見つけるのは難しかったと思う。
・Kより大きい小さいで2値化、の視点はまあ合ってた。
・Kを広げていくだけでなく、Kより大きい値を寄せてきて最終的にKと繋げる、
　っていう発想は自分からは出なかったぁ。(割とみんな同じこと言ってたけど)
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
    N, K = MAP()
    A = LIST()

    if not A.count(K):
        print('no')
        continue
    
    if N == 1 and A[0] == K:
        print('yes')
        continue

    B = [0] * N
    for i in range(N):
        if A[i] >= K:
            B[i] = 1
        else:
            B[i] = -1
    acc = [-INF] + list(accumulate(B)) + [-INF]
    accrev = [-INF] + list(accumulate(B[::-1]))[::-1] + [-INF]

    st = SegTree(N+2, max, -INF, acc)
    strev = SegTree(N+2, max, -INF, accrev)
    for i in range(N):
        if A[i] == K:
            if strev.query(0, i+1) > 0 or st.query(i+2, N+2) > 0:
                print('yes')
                break
    else:
        print('no')
