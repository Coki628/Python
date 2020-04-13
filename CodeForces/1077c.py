# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・セグ木使って最大値を動的に持てれば、後は合計も合わせて動かして
　1箇所ずつ愚直に試していける。
・20万にセグ木のlog乗ってるけど、pypyAC0.4秒。
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

N = INT()
A = LIST()
st = SegTree(N, max, 0, A)

sm = sum(A)
ans = []
for i, a in enumerate(A):
    st.update(i, 0)
    mx = st.all()
    sm -= a
    if sm - mx == mx:
        ans.append(i+1)
    st.update(i, a)
    sm += a
print(len(ans))
print(*ans)
