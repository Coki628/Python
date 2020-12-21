# -*- coding: utf-8 -*-

"""
・蟻本演習2-5-1、自力ならず。
・二部グラフ判定
・やったことあって、テーマが分かってて、それでもこれ気付けなかったのつらい。。
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

N, M = MAP()
nodes = tuple([] for i in range(N))
for i in range(M):
    a, b = MAP()
    a -= 1
    b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

# 二部グラフ判定
color = [-1] * N
def rec(u, prev, c):
    if color[u] == -1:
        color[u] = c
    else:
        if color[u] == c:
            return True
        else:
            return False
    for v in nodes[u]:
        if v != prev:
            if not rec(v, u, 1-c):
                return False
    return True

if rec(0, -1, 0):
    a = sum(color)
    b = N - a
    ans = a*b - M
else:
    ans = N*(N-1) // 2 - M
print(ans)
