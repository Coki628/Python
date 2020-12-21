# -*- coding: utf-8 -*-

"""
・自力AC、確率系ちゃんと通せたのは嬉しい。
・確率系
・aが取る値を決め打つことで、その値でb,cにどのくらい負けるかが分かるので、
　後は足し合わせればOK。
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

al, ar = MAP()
BC = [[]] * 2
BC[0] = LIST()
BC[1] = LIST()

ans = 0
totala = ar - al + 1
# aが取りうる範囲について全探索
for a in range(al, ar+1):
    cur = 1
    for l, r in BC:
        total = r - l + 1
        # b(c)が取りうる範囲全体の中で、今回のaが負ける回数
        losecnt = min(max(r - a, 0), total)
        cur *= losecnt / total
    ans += cur / totala
print(ans)
