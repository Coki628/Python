# -*- coding: utf-8 -*-

"""
・実際内ループはlogで済むので愚直シミュレーションでOK。
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

N = 2**INT()
A = LIST()

ans = [1] * N
cur = [(i, a) for i, a in enumerate(A)]
while len(cur) > 2:
    nxt = []
    for i in range(0, len(cur), 2):
        if cur[i][1] > cur[i+1][1]:
            ans[cur[i][0]] += 1
            nxt.append(cur[i])
        else:
            ans[cur[i+1][0]] += 1
            nxt.append(cur[i+1])
    cur = nxt
[print(a) for a in ans]
