"""
・さくっと自力AC
・グラフ、BFS
・出発点から、1つ前の頂点を覚えながら進めばいい。
　グラフは連結が保証されてるのでNoはなし。
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

def bfs(nodes, src):
    from collections import deque

    N = len(nodes)
    que = deque([src])
    res = [-1] * N
    res[src] = 0
    while que:
        u = que.popleft()
        for v in nodes[u]:
            if res[v] != -1:
                continue
            res[v] = u
            que.append((v))
    return res

N, M = MAP()
nodes = [[] for i in range(N)]
for i in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

res = bfs(nodes, 0)
Yes()
for a in res[1:]:
    print(a+1)
