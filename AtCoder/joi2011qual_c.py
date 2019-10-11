# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-14、さくっと自力AC
・ソートしてからナップサックDPって欄で紹介されてたけど、これはソートして貪欲では。。
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
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
A, B = MAP()
C = INT()
D = LIST(N)

D.sort(reverse=True)
cal = 0
cost = A
mx = (C+cal) // cost
for i in range(N):
    cal += D[i]
    mx = max(mx, (C+cal)//(cost+B*(i+1)))
print(mx)
