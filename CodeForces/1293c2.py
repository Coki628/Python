# -*- coding: utf-8 -*-

"""
・自力AC
・BITで通れなくなっている列を管理する。
・特に再度通れるようになった時の確認が、条件足りてるか超自信なかったけど、システス通った。
・BITクラスをちょっと整備。1点更新しやすいようにした。
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

class BIT:

    def __init__(self, n):
        nv = 1
        n += 1
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

N, Q = MAP()
grid = list2d(2, N, 0)
bit = BIT(N)

for _ in range(Q):
    r, c = MAP()
    r -= 1; c -= 1
    # マスの切り替え
    grid[r][c] ^= 1
    # 通れなくなった時
    if grid[r][c]:
        # 1つ前と今の位置の関係から確認
        if c != 0 and grid[r^1][c-1]:
            bit.update(c, 1)
        # 今の位置上下の確認
        if grid[r^1][c]:
            bit.update(c, 1)
        # 今の位置と1つ先の関係から確認
        if c != N-1 and grid[r^1][c+1]:
            bit.update(c+1, 1)
    # 通れるようになった時
    else:
        # 1つ前と今の位置の関係から確認
        if c != 0 and not (grid[r^1][c] & grid[r][c-1]) and not (grid[r^1][c-1] & grid[r][c]):
            bit.update(c, 0)
        # 今の位置と1つ先の関係から確認
        if c != N-1 and not (grid[r^1][c] & grid[r][c+1]) and not (grid[r^1][c+1] & grid[r][c]) and not (grid[r^1][c+1] & grid[r][c+1]):
            bit.update(c+1, 0)
    # 最後まで通れるならOK
    if bit.sum(N) == 0:
        Yes()
    else:
        No()
