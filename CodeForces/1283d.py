# -*- coding: utf-8 -*-

"""
・自力AC！
・BFS応用
・数直線上で多点スタートのBFSを走らせる。
・範囲が広いので、距離の管理はdictでやる。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def bfs(src):
    from collections import deque

    que = deque()
    for x in src:
        que.append((x, 0))
    dist = {}
    while len(dist) < N+M:
        u, c = que.popleft()
        if u in dist:
            continue
        dist[u] = c
        que.append((u-1, c+1))
        que.append((u+1, c+1))
    return dist

N, M = MAP()
A = LIST()

res = bfs(A)
ans = []
sm = 0
for k, v in res.items():
    if v != 0:
        sm += v
        ans.append(k)
print(sm)
print(*ans)
