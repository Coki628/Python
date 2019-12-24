# -*- coding: utf-8 -*-

"""
・AC後、ちょっと整理した。
・DFS2回、木の最短経路
・最終的にはまあ、考察もほぼ公式解通りだし良かった。
・後はもっと手早くこの形にまとめられればよかったんだけど。。
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

def dfs(x):
    dist  = [INF] * N
    stack = [(x, 0, -1)]
    while stack:
        u, cost, prev = stack.pop()
        if dist[u] != INF:
            continue
        dist[u] = cost
        for v in nodes[u]:
            if v == prev:
                continue
            stack.append((v, cost+1, u))
    return dist

# 追われる側、追う側それぞれからの最短距離
res1 = dfs(x)
res2 = dfs(y)

mx = -INF
for i in range(N):
    # 追われる側のが先に着く場所は逃げ込む候補になりうる
    if res1[i] <= res2[i]:
        # その中で一番追う側から遠い場所
        mx = max(mx, res2[i])
# 最終的に1歩手前で捕まるので-1
print(mx - 1)
