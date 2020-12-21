# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/agc033/editorial.pdf
・木上でゲーム系
・木の直径
・状況が1か2ずつしか動かせないのは分かったのに、葉の数とか見てて
　木の直径に辿り着けなかった。。これは悔しい。
・20万のNにスタックDFSでpythonAC1.0秒、pypyAC0.7秒。
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
INF = 10 ** 19
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
for _ in range(N-1):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

# 木の直径
dist1 = dfs(N, nodes, 0)
u = dist1.index(max(dist1))
dist2 = dfs(N, nodes, u)
L = max(dist2)

if L % 3 != 1:
    print('First')
else:
    print('Second')
