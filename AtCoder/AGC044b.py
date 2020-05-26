"""
・自力ならず。
・グリッド、01BFS
・逆から見ていったけどこれは失敗。多分N^4してる。TLE。
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

def bfs(src):
    from collections import deque

    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    que = deque()
    h, w = src
    que.append((0, h, w))
    visited = list2d(N, N, 0)
    visited[h][w] = 1
    while que:
        cost, h, w = que.popleft()
        for dh, dw in directions:
            h2 = h + dh
            w2 = w + dw
            if h2 < 0 or w2 < 0 or h2 >= N or w2 >= N:
                h, w = src
                dist[h][w] = cost
                return cost
            if visited[h2][w2]:
                continue
            visited[h2][w2] = 1
            if dist[h2][w2] == -1:
                que.appendleft((cost, h2, w2))
            else:
                que.append((cost+1, h2, w2))

N = INT()
P = [p-1 for p in LIST()]

dist = list2d(N, N, -1)
ans = 0
for p in P[::-1]:
    h, w = p//N, p%N
    ans += bfs((h, w))
print(ans)
