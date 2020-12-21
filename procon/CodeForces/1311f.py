# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/contest/1311/submission/71799062
・ほぼ写経fromじゅぴろさん
・BIT2つ使う。強い人には典型みたい。
"""

import sys
from collections import Counter

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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

class BIT:

    def __init__(self, n):
        n += 1
        nv = 1
        while nv < n:
            nv *= 2
        self.size = nv
        self.tree = [0] * nv

    def sum(self, i):
        s = 0
        i += 1
        while i > 0:
            s += self.tree[i-1]
            i -= i & -i
        return s

    def add(self, i, x):
        i += 1
        while i <= self.size:
            self.tree[i-1] += x
            i += i & -i

    def get(self, l, r=None):
        if r is None: r = l + 1
        res = 0
        if r: res += self.sum(r-1)
        if l: res -= self.sum(l-1)
        return res

    def update(self, i, x):
        self.add(i, x - self.get(i))

def compress(S):

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(S)):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

N = INT()
A = LIST()
B = LIST()

BA = list(zip(B, A))
BA.sort()

zipped, _ = compress(set(A))

bit1 = BIT(N)
bit2 = BIT(N)
ans = 0
for b, a in BA:
    sm = bit1.sum(zipped[a])
    cnt = bit2.sum(zipped[a])
    ans += cnt * a - sm
    bit1.add(zipped[a], a)
    bit2.add(zipped[a], 1)
print(ans)
