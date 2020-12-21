"""
・さくっと自力AC
・BFS
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
EPS = 10 ** -10

def popcount(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0f0f0f0f
    i = i + (i >> 8)
    i = i + (i >> 16)
    return i & 0x3f

def bfs(src, N):
    """ BFS(一般グラフ、重みなし) """
    from collections import deque

    que = deque([src])
    dist = [INF] * (N+1)
    dist[src] = 1
    while que:
        u = que.popleft()
        v = u + popcount(u)
        if v <= N and dist[v] == INF:
            dist[v] = dist[u] + 1
            que.append(v)
        v = u - popcount(u)
        if v >= 1 and dist[v] == INF:
            dist[v] = dist[u] + 1
            que.append(v)
    return dist

N = INT()

res = bfs(1, N)
ans = res[N]
if ans == INF:
    print(-1)
else:
    print(ans)
