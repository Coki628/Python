"""
参考：https://atcoder.jp/contests/agc044/submissions/13596168
・4方向のループをif文4つにしたらAC。こういう定数倍高速化は本当はやりたくないな。。可読性下がるし。。
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

    que = deque()
    h, w = src
    que.append((dist[h][w], h, w))
    while que:
        cost, h, w = que.popleft()
        if h > 1:
            if dist[h-1][w] > cost:
                dist[h-1][w] = cost
                que.append((cost+seated[h-1][w], h-1, w))
        if h < N-1:
            if dist[h+1][w] > cost:
                dist[h+1][w] = cost
                que.append((cost+seated[h+1][w], h+1, w))
        if w > 1:
            if dist[h][w-1] > cost:
                dist[h][w-1] = cost
                que.append((cost+seated[h][w-1], h, w-1))
        if w < N-1:
            if dist[h][w+1] > cost:
                dist[h][w+1] = cost
                que.append((cost+seated[h][w+1], h, w+1))

N = INT()
P = [p-1 for p in LIST()]

dist = list2d(N, N, 0)
for i in range(N):
    for j in range(N):
        dist[i][j] = min(i, N-i-1, j, N-j-1)

ans = 0
seated = list2d(N, N, 1)
for p in P:
    h, w = divmod(p, N)
    ans += dist[h][w]
    bfs((h, w))
    seated[h][w] = 0
print(ans)
