# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-6、自力AC！
・LIS応用、気づいてしまえば実装はシンプル。
・ただしこれは「LISが関係してる」っていうテーマが前提としてあって解いたからできたようなもので、
　ノーヒントでLISに辿り着けたかどうかは分からない。。
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

def lenLIS(A: list, equal=False):
    """ 最長増加部分列の長さ """
    from operator import gt, ge
    from bisect import bisect_left, bisect_right

    # デフォルトは狭義のLIS(同値を含まない)
    compare = gt if not equal else ge
    bisect = bisect_left if not equal else bisect_right
    L = [A[0]]
    for a in A[1:]:
        if compare(a, L[-1]):
            L.append(a)
        else:
            L[bisect(L, a)] = a
    return len(L)

N = INT()
A = []
for i in range(N):
    x, r = MAP()
    A.append((x-r, x+r))

A.sort()
L, R = zip(*A)
print(lenLIS(R[::-1]))
