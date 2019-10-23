# -*- coding: utf-8 -*-

"""
・自力AC
・これはRE。多分、再帰が深すぎる。
・C++で同じやつ書いてのAC。
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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def rec(i, org, cnt):
    nxt = A[i]
    if nxt != org:
        cnt = rec(nxt, org, cnt+1)
    ans[i] = cnt
    return cnt

for _ in range(INT()):
    N = INT()
    A = [a-1 for a in LIST()]
    ans = [-1] * N
    for i in range(N):
        if ans[i] == -1:
            rec(i, i, 1)
    print(*ans)
