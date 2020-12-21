"""
・自力AC
・一般グラフ、BFS応用
・基本的には連結成分毎の最短経路を取る。
・次に友達になる条件がちょっと独特で、やってみると2冪くらいの速度で繋がって行くのが分かる。
・10万頂点のグラフで30クエリを捌くんだけど、BFSのINFを-1にしたら1.86→0.99秒まで速くなった。
　やっぱ値がでかいと配列の生成コストが馬鹿にならんのだなー。
"""

import sys
from math import log2

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
EPS = 10 ** -10

def bfs(nodes, src):
    """ BFS(一般グラフ、重みなし) """
    from collections import deque

    N = len(nodes)
    que = deque([src])
    dist = [-1] * N
    dist[src] = 0
    cnt = 0
    while que:
        u = que.popleft()
        for v in nodes[u]:
            if dist[v] != -1:
                continue
            dist[v] = dist[u] + 1
            que.append(v)
            cnt += 1
    mx = 0
    for i in range(N):
        if dist[i] != -1 and dist[i] >= 2:
            mx = max(mx, ceil(log2(dist[i])))
    return cnt, mx

N, M = MAP()
nodes = [[] for i in range(N)]
for i in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

Q = INT()
for _ in range(Q):
    x = INT()
    x -= 1

    cnt, mx = bfs(nodes, x)
    print(cnt, mx)
