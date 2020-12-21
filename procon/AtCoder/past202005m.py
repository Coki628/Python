"""
・BFSとTSP(巡回セールスマン)
・前に似たようなの(jag2013autumn_c)やったことある。
・グラフ自体は大きいけど、結局メインで使うのは通る10数箇所だけ。
・もう後半で疲れてたのもあって、元グラフとTSPで頂点番号紐付けるのに神経使った。。
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

def bfs(nodes, src):
    from collections import deque

    N = len(nodes)
    que = deque([src])
    dist = [INF] * N
    dist[src] = 0
    while que:
        u = que.popleft()
        for v in nodes[u]:
            if dist[v] != INF:
                continue
            dist[v] = dist[u] + 1
            que.append(v)
    return dist

N, M = MAP()
nodes = [[] for i in range(N)]
for i in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)
s = INT()
s -= 1
K = INT()
K += 1
T = [s] + [t-1 for t in LIST()]
atot = {}
ttoa = {}
for i, t in enumerate(T):
    atot[t] = i
    ttoa[i] = t

dist = [[]] * K
for i in range(K):
    dist[i] = bfs(nodes, ttoa[i])

dp = list2d(1<<K, K, INF)
dp[1<<atot[s]][atot[s]] = 0
for bit in range(1, (1<<K)-1):
    for i in range(K):
        if not (bit>>i & 1):
            continue
        for j in range(K):
            if bit>>j & 1:
                continue
            dp[bit|1<<j][j] = min(dp[bit|1<<j][j], dp[bit][i] + dist[i][ttoa[j]])
ans = min(dp[-1])
print(ans)
