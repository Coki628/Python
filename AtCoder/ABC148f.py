# -*- coding: utf-8 -*-

"""
・コンテスト終了約30分後、自力AC。
・Eに時間取られ過ぎた。。
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

N, x, y = MAP()
x -= 1; y -= 1
nodes = [[] for i in range(N)]
for i in range(N-1):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

memo  = [INF] * N
stack = [(y, 0, -1)]
while stack:
    u, cost, prev = stack.pop()
    if memo[u] != INF:
        continue
    memo[u] = cost
    for v in nodes[u]:
        if v == prev:
            continue
        stack.append((v, cost+1, u))
memo2  = [INF] * N
stack = [(x, 0, -1)]
while stack:
    u, cost, prev = stack.pop()
    if memo2[u] != INF:
        continue
    memo2[u] = cost
    if u == y:
        continue
    for v in nodes[u]:
        if v == prev:
            continue
        stack.append((v, cost+1, u))

mx = -INF
for i in range(N):
    if memo[i] >= memo2[i]:
        mx = max(mx, memo[i])
print(mx - 1)
