# -*- coding: utf-8 -*-

"""
・LISがにぶたんじゃなくてセグ木でも出せると聞いてやってみた。
・書いてみると、ああ確かにー、って感じ。
・速度は同じlogだけどにぶたん(bisect)より全然遅かった。。
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
    """ BIT汎用版 """

    def __init__(self, n, func, init):
        # 0-indexed
        n += 1
        nv = 1
        while nv < n:
            nv *= 2
        self.size = nv
        self.func = func
        self.init = init
        self.tree = [init] * nv
    
    def query(self, i):
        """ [0, i]の値を取得 """
        s = self.init
        i += 1
        while i > 0:
            s = self.func(s, self.tree[i-1])
            i -= i & -i
        return s
    
    def update(self, i, x):
        """ 値の更新：添字i, 値x """
        i += 1
        while i <= self.size:
            self.tree[i-1] = self.func(self.tree[i-1], x)
            i += i & -i

N = INT()
A = [a-1 for a in LIST()]

bit = BIT(N, max, 0)
for a in A:
    # 自分より手前の最大 + 1
    bit.update(a, bit.query(a) + 1)
ans = bit.query(N-1)
print(ans)
