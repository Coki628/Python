# -*- coding: utf-8 -*-

"""
・自力AC
・再帰探索(メモ化再帰)
・N<=30と小さかったので、条件を満たす数列をタプルにしてsetで出現済管理して、
　条件を満たす組み方を全部探しに行った。
・数列自体をキーにするのは重くないかちょっと心配だったけど、
　pythonで0.1秒かからずAC。
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

N = INT()
A = tuple([INF] + [1] * N)

memo = set()
memo.add(A)
def rec(A):
    for i in range(1, len(A)-1):
        if A[i-1] >= A[i] + A[i+1]:
            nxt = A[:i] + (A[i] + A[i+1],) + A[i+2:]
            # 未出現の数列だったら、そこからさらに作れるか探しに行く
            if nxt not in memo:
                memo.add(nxt)
                rec(nxt)
rec(A)
ans = list(memo)
# 番兵INFを削る
ans = [a[1:] for a in ans]
ans.sort(reverse=1)
[print(*a) for a in ans]
