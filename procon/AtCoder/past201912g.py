# -*- coding: utf-8 -*-

"""
・きっちり自力AC
・全探索
・最大で10人を3グループ=3^10=約6万に、*2人組の組み合わせだからpythonだとどうかなーと思ったけど、
　pythonAC0.6秒、pypyAC0.3秒。
"""

import sys
from itertools import product, combinations

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

N = INT()
table = list2d(N, N, 0)
ln = N - 1
for i in range(N-1):
    li = LIST()
    for j in range(ln):
        k = i + j + 1
        table[i][k] = li[j]
        table[k][i] = li[j]
    ln -= 1

ans = -INF
for x in [1, 2, 3]:
    # N人がx個のグループに分かれる(x^N)
    for prod in product(range(x), repeat=N):
        sm = 0
        groups = [[] for i in range(x)]
        for i, g in enumerate(prod):
            groups[g].append(i)
        for group in groups:
            # グループ内での2人組を全組み合わせ(最大でnCr(N, 2))
            for comb in combinations(group, 2):
                a, b = comb
                sm += table[a][b]
        ans = max(ans, sm)
print(ans)
