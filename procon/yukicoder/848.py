"""
・なんとか自力AC、危なかったー。。
・ダイクストラ
・最短経路求めて、中継地点を決めて全探索、は結構すぐやったんだけどWA。。
　すげー色々悩んだ後、制約がN^2を許す意味をよくよく考えて、
　「もしかして行きと帰りで別の中継地を使った方が得する場合があるのでは」
　という考えに行き着く。
・結果的にはこれを正しく適用したら無事AC。
・でも無向グラフで往復のコスト変わらないのに、同じ中継地じゃダメなの直感に反するなー。。
　ってこれ書きながら経路のイメージしてたら、何か別の中継地のが早く合流できるパターンが
　全然ありそうな気がしてきた。知ってからだと考え変わるよね。。
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

N, M, p, q, t = MAP()
p -= 1; q -= 1
nodes = [[] for i in range(N)]
for i in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1

    nodes[a].append((b, c))
    nodes[b].append((a, c))

dist1 = dijkstra(nodes, 0)
dist2 = dijkstra(nodes, p)
dist3 = dijkstra(nodes, q)

ans = -1
for i in range(N):
    for j in range(N):
        # 途中で別れてP,Qをそれぞれに行く
        have = t - (dist1[i] + dist1[j] + max(dist2[i] + dist2[j], dist3[i] + dist3[j]))
        if have >= 0:
            ans = max(ans, have + dist1[i] + dist1[j])
        # 2人で一緒にP,Qに行く
        have = t - (dist1[i] + dist2[i] + dist3[i] + dist1[j] + dist2[j] + dist3[j])
        if have >= 0:
            ans = max(ans, t)
print(ans)
