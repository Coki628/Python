"""
・危なかった。なんとか自力AC。
・ダイクストラ
・また中継地全探索かなー、とかって始点と終点からダイクストラしたけどこれがハズレ。
・正解は、ワープゾーンの分の頂点を増やす、だった。まあ、頂点拡張だよね。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

N, M = MAP()
AB, AC, BC = MAP()
D = {
    'AB': AB, 'BA': AB, 'AC': AC, 'CA': AC, 'BC': BC, 'CB': BC,
    'AA': INF, 'BB': INF, 'CC': INF,
}
S = input()
nodes = [[] for i in range(N+3)]
for i in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    nodes[a].append((b, c))
    nodes[b].append((a, c))

for i in range(N):
    if S[i] == 'A':
        # Aのワープゾーンに入る
        nodes[i].append((N+1, D['AB']))
        nodes[i].append((N+2, D['AC']))
        # Aのワープゾーンから出てくる
        nodes[N].append((i, 0))
    elif S[i] == 'B':
        nodes[i].append((N, D['BA']))
        nodes[i].append((N+2, D['BC']))
        nodes[N+1].append((i, 0))
    else:
        nodes[i].append((N, D['CA']))
        nodes[i].append((N+1, D['CB']))
        nodes[N+2].append((i, 0))

s = 0
t = N - 1
dist = dijkstra(nodes, s)
ans = dist[t]
print(ans)
