# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/73953
　　　https://twitter.com/butsurizuki/status/1229545439164170247
　　　https://twitter.com/armeria_betrue/status/1229471643799441408
・自力ならず。
・グラフ、最短経路応用
・始点、終点両方からBFSで最短距離を取っておく。(これはできた)
・任意の(i,j)(i≠j)の組について、min(A_i+B_j,A_j+B_i)の最大値は、A_i+B_j <= A_j+B_iを移行してA_i-B_i <= A_j-B_j
　これをA_k-B_kの昇順にソートすると、相対的になるべくAが小さくてBが大きい感じのものから並ぶ。
  この中で(i<j)についてA[i]とペアになって最大値を作るのはmax(B[i+1:N])となる。
・実装上は、逆にB[i]とペアになって最大値を作るのはmax(A[0:i-1])、を使って、
　Aを累積maxしながらBを1つずつ見ていく、とやることですっきりまとまる、をみんなやっていた。
・なんか強い人には割と有名テク？みたいだ。
・計算量は20万のNにソートのlogが乗るだけのはずなんだけど、
　なんかpypyでも遅くて最初はAC1.6秒。
・気になったからちょっと速度改善入れて、
　・ソートのキー指定やる → 1.4秒
　・集合管理をsetからlistのindexに → 1.3秒
・それでもこの程度だから、多分20万頂点20万辺への最短経路BFS2回がきっと重いのだろう。
"""

import sys
from operator import itemgetter

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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def bfs(N, nodes, src):
    from collections import deque

    que = deque([(src, 0)])
    dist = [INF] * N
    while que:
        u, c = que.popleft()
        if dist[u] != INF:
            continue
        dist[u] = c
        for v in nodes[u]:
            que.append((v, c+1))
    return dist

N, M, K = MAP()
S = [0] * N
for a in LIST():
    S[a-1] = 1
nodes = [[] for i in range(N)]
for _ in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

# 始点、終点両方から最短距離を取っておく
dist1 = bfs(N, nodes, 0)
dist2 = bfs(N, nodes, N-1)
mn = dist1[N-1]

AB = []
for i in range(N):
    a, b = dist1[i], dist2[i]
    if S[i]:
        AB.append((a-b, a, b))
# a-bの昇順でソート
AB.sort(key=itemgetter(0))
_, A, B = zip(*AB)
ans = 0
accmx = A[0]
for i in range(1, K):
    a, b = A[i], B[i]
    # B[i]とペアで最大となるのはi-1までのAの累積max
    ans = max(ans, accmx + b + 1)
    accmx = max(accmx, a)
ans = min(ans, mn)
print(ans)
