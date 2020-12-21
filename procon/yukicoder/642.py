"""
・自力AC
・グラフに帰着、整数BFS
・最初、順操作でやったらめっちゃ時間かかってたので、逆操作にしてみたら一瞬だった。
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
EPS = 10 ** -10

def bfs(src):
    """ BFS(一般グラフ、重みなし) """
    from collections import deque

    que = deque([src])
    dist = {}
    dist[src] = 0
    while que:
        u = que.popleft()
        if u == 1:
            return dist[u]
        if u & 1 == 0:
            v = u // 2
            if v in dist:
                continue
            dist[v] = dist[u] + 1
            que.append(v)
        v = u + 1
        if v in dist:
            continue
        dist[v] = dist[u] + 1
        que.append(v)
    return -1

N = INT()

res = bfs(N)
print(res)
