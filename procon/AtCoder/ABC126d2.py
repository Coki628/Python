# -*- coding: utf-8 -*-

"""
・蟻本演習2-5-1、さくっと自力AC
・二部グラフ
・つい最近やった、親への辺のコストを見るっていう方針が使えた。
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
nodes = tuple([] for i in range(N))
for i in range(N-1):
    a, b, c = MAP()
    a -= 1
    b -= 1
    nodes[a].append((b, c))
    nodes[b].append((a, c))

ans = [0] * N
def rec(u, prev, prevc):
    # 親への辺のコストによって塗り分け
    if prev == -1:
        ans[u] = 0
    elif prevc % 2 == 0:
        ans[u] = ans[prev]
    else:
        ans[u] = 1 - ans[prev]
    for v, c in nodes[u]:
        if v != prev:
            rec(v, u, c)
rec(0, -1, 0)

[print(a) for a in ans]
