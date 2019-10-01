# -*- coding: utf-8 -*-

"""
・色々頑張って自力実装したけどTLE
・BIT二分探索を使ってみた。少し遅くなった。。
"""

import sys
from collections import defaultdict

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

class BIT:

    def __init__(self, n):
        # 0-indexed
        nv = 1
        while nv < n:
            nv *= 2
        self.size = nv
        self.tree = [0] * nv

    def sum(self, i):
        """ [0, i]を合計する """
        s = 0
        i += 1
        while i > 0:
            s += self.tree[i-1]
            i -= i & -i
        return s

    def add(self, i, x):
        """ 値の追加：添字i, 値x """
        i += 1
        while i <= self.size:
            self.tree[i-1] += x
            i += i & -i

    def get(self, l, r=None):
        """ 区間和の取得 [l, r) """
        # 引数が1つなら一点の値を取得
        if r is None: r = l + 1
        res = 0
        if r: res += self.sum(r-1)
        if l: res -= self.sum(l-1)
        return res

N, Q = MAP()
ST = set()
X = set()
A = defaultdict(list)
for i in range(N):
    s, t, x = MAP()
    # 全ての工事を同じ時系列に持ってくる
    ST.add(s-x)
    ST.add(t-x)
    X.add(x)
    A[s-x].append((x, 0))
    A[t-x].append((x, 1))
D = LIST(Q)

# 座標圧縮
zip_x, unzip_x = {}, {}
X = sorted(X)
for i in range(len(X)):
    zip_x[X[i]] = i
    unzip_x[i] = X[i]

def calc(m):
    cnt = bit.sum(m)
    return cnt > 0

Dset = set(D)
S = sorted(ST|Dset)
B = {}
bit = BIT(len(X))
for key in S:
    li = A[key]
    for x, flag in li:
        if flag == 0:
            bit.add(zip_x[x], 1)
        else:
            bit.add(zip_x[x], -1)
    if key in Dset:
        # BIT二分探索
        mn = bisearch_min(-1, len(X), calc)
        B[key] = unzip_x[mn] if mn != len(X) else -1
[print(B[x]) for x in D]
