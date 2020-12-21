# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC
・木の最短経路、DFS2回。
・木上の2人ゲームで両方から最短経路取るのは、この前ABC148fで出たのと同じ方針だね。
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

def dfs(N, nodes, src):
    """ DFS(木、スタック、重みなし) """

    stack = [(src, -1, 0)]
    dist = [INF] * N
    while stack:
        u, prev, c = stack.pop()
        dist[u] = c
        for v in nodes[u]:
            if v != prev:
                stack.append((v, u, c+1))
    return dist

N = INT()
nodes = [[] for i in range(N)]
for i in range(N-1):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

# それぞれから最短距離
res1 = dfs(N, nodes, 0)
res2 = dfs(N, nodes, N-1)

cnt = 0
for i in range(N):
    # 相手より距離が近い場所(同値含む)は自分のマスにできる
    if res1[i] <= res2[i]:
        cnt += 1
# 自分のマスが半分超なら勝ち(半分ちょうどだと負ける)
if cnt > N // 2:
    print('Fennec')
else:
    print('Snuke')
