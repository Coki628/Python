# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・木の直径
・計算量は10万頂点の木で、pythonAC0.4秒。
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
nodes = [[] for i in range(N)]
for i in range(N-1):
    u, v = MAP()
    u -= 1; v -= 1
    nodes[u].append(v)
    nodes[v].append(u)

def dfs(start):
    stack = [(start, -1, 0)]
    memo = [INF] * N
    while stack:
        u, prev, dist = stack.pop()
        memo[u] = dist
        dist += 1
        for v in nodes[u]:
            if v != prev:
                stack.append((v, u, dist))
    return memo

# 木の直径
res = dfs(0)
x = res.index(max(res))
res = dfs(x)
y = res.index(max(res))
print(x+1, y+1)
