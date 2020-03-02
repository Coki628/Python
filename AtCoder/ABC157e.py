# -*- coding: utf-8 -*-

"""
・結構すんなり自力AC！
・BIT26本生やして各文字毎に出現位置を管理する。
・前これのセグ木版でほぼ似たようなことやったことがあって良かった。
・計算量、クエリは2万だけど、配列が50万なので50万にlogが乗る。
　pythonTLE,pypyAC1.0秒。
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

class BIT:
    """ Binary Indexed Tree """

    def __init__(self, n):
        # 0-indexed
        n += 1
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

    def update(self, i, x):
        """ 値の更新：添字i, 値x """
        self.add(i, x - self.get(i))

N = INT()
S = [ord(c)-97 for c in input()]
BITs = [BIT(N+1) for i in range(26)]
for i, s in enumerate(S):
    BITs[s].add(i, 1)

Q = INT()
for _ in range(Q):
    q = list(input().split())
    if q[0] == '1':
        _, i, c = q
        i = int(i) - 1
        c = ord(c) - 97
        if S[i] != c:
            prev = S[i]
            S[i] = c
            BITs[prev].add(i, -1)
            BITs[c].add(i, 1)
    else:
        _, l, r = q
        l, r = int(l) - 1, int(r) - 1
        ans = 0
        for c in range(26):
            if BITs[c].get(l, r+1) > 0:
                ans += 1
        print(ans)
