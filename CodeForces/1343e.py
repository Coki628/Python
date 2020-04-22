# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/momohara_kyopro/status/1252637091319574528
　　　https://twitter.com/laycrs/status/1252637433260175360
　　　https://twitter.com/titia_til/status/1252640073582927872
・自力ならず。
・グラフ、最短経路、BFS、経由地を全探索
・考え方はpast201912jと同じ感じ。使う経由地を全部試す。
・例えば、x-経由 と y-経由 の両方に出てくる辺があってコスト割り当てにバグったりしないのかとか
　考えたんだけど、最適な経由地を選ぶ時にはそういう重複は発生しない、って感じなんだと思う、多分。。
"""

import sys
from itertools import accumulate

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

def bfs(nodes, src):
    from collections import deque

    N = len(nodes)
    que = deque([(src, 0)])
    dist = [INF] * N
    dist[src] = 0
    while que:
        u, c = que.popleft()
        for v in nodes[u]:
            if dist[v] != INF:
                continue
            dist[v] = c + 1
            que.append((v, c+1))
    return dist

out = []
for _ in range(INT()):
    N, M, x, y, z = MAP()
    x -= 1; y -= 1; z -= 1
    cost = LIST()
    nodes = [[] for i in range(N)]
    for _ in range(M):
        a, b = MAP()
        a -= 1; b -= 1
        nodes[a].append(b)
        nodes[b].append(a)

    cntx = bfs(nodes, x)
    cnty = bfs(nodes, y)
    cntz = bfs(nodes, z)

    cost.sort()
    cost = [0] + list(accumulate(cost))
    ans = INF
    for i in range(N):
        total = cntx[i] + cnty[i] + cntz[i]
        # 辺がたくさん重なってM本を超える場合もあるのでそれは除外
        if total > M:
            continue
        # yへの経路だけ2回通るので2回分足す
        ans = min(ans, cost[total] + cost[cnty[i]])
    out.append((ans))
print('\n'.join(map(str, out)))
