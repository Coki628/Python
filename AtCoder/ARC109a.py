"""
・さくっと自力AC
・グラフに帰着、ダイクストラ
・頭いい場合分け解法ありそうと思ったけど、そんなん正確に見切れるほど賢くないので、
　迷わず辺張ってダイクストラ貼ってえいってやる。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def dijkstra(nodes: list, src: int) -> list:
    """ ダイクストラ高速化版(隣接リスト(0-indexed), 始点) """
    from heapq import heappush, heappop

    N = len(nodes)
    res = [INF] * N
    que = [src]
    res[src] = 0
    while que:
        cur = heappop(que)
        dist = cur // N
        cur %= N
        if res[cur] < dist:
            continue
        for nxt, cost in nodes[cur]:
            if dist + cost < res[nxt]:
                res[nxt] = dist + cost
                heappush(que, (dist+cost)*N+nxt)
    return res

a, b, x, y = MAP()

N = 100
M = 100

nodes = [[] for i in range(N+M)]
for i in range(N):
    nodes[i].append((i+N, x))
    nodes[i+N].append((i, x))
for i in range(N-1):
    nodes[i+1].append((i+N, x))
    nodes[i+N].append((i+1, x))
for i in range(N-1):
    nodes[i].append((i+1, y))
    nodes[i+1].append((i, y))
    nodes[N+i].append((N+i+1, y))
    nodes[N+i+1].append((N+i, y))

a -= 1
b -= 1
b += N
res = dijkstra(nodes, a)
ans = res[b]
print(ans)
