# -*- coding: utf-8 -*-

"""
・サンプルでWA
・先にどっちか作って、その状態からの最短経路とか考えて色々やってみたが、うまくいかず。
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
INF = float('inf')
MOD = 10 ** 9 + 7

def deepcopy(li): return [x[:] for x in li]

def dijkstra(N: int, nodes: list, src: int) -> list:
    """ ダイクストラ高速化版(頂点数, 隣接リスト(0-indexed), 始点) """
    from heapq import heappush, heappop

    # 頂点(ある始点からの最短距離, 前の頂点)
    res = [(INF, -1) for i in range(N)]
    # スタート位置
    que = [src]
    res[src] = (0, -1)
    # キューが空になるまで
    while len(que) != 0:
        # 距離*N + 現在のノード
        cur = heappop(que)
        # 距離とノードに分ける
        dist = cur // N
        cur %= N
        # 出発ノードcurの到着ノードでループ
        for nxt, cost in nodes[cur]:
            # 今回の経路のが短い時
            if dist + cost < res[nxt][0]:
                res[nxt] = (dist+cost, cur)
                # 距離*N+ノード番号 の形でキューに詰める
                heappush(que, (dist+cost)*N+nxt)
    # ノードsrcからの最短距離と経路のリストを返却
    return res

# s,t間の経路を取得
def get_route(s, t, res):
    prev = t
    StoT = [t]
    while prev != s:
        prev = res[prev][1]
        if prev == -1:
            return None
        StoT.append(prev)
    StoT = StoT[::-1]
    return StoT

N, M = MAP()
w, x, y, z = MAP()
w -= 1
x -= 1
y -= 1
z -= 1
nodes = [[] for i in range(N)]
edges = {}
for i in range(M):
    c, s, t = MAP()
    nodes[s-1].append((t-1, c))
    edges[(s-1, t-1)] = c

fromW = dijkstra(N, nodes, w)
fromY = dijkstra(N, nodes, y)

WtoX = get_route(w, x, fromW)
WtoZ = get_route(w, z, fromW)
YtoX = get_route(y, x, fromY)
YtoZ = get_route(y, z, fromY)

ans = INF

if not WtoX or not YtoZ:
    print('Impossible')
    exit()

# WtoX => YtoZ
edges2 = edges.copy()
nodes2 = [[] for i in range(N)]
for i in range(len(WtoX)-1):
    edges2[(WtoX[i], WtoX[i+1])] = 0
for k, v in edges2.items():
    s, t = k
    c = v
    nodes2[s].append((t, c))
fromY2 = dijkstra(N, nodes2, y)
# YtoZ2 = get_route(y, z, fromY2)
# common = set(WtoX) & set(YtoZ2)
# if len(common):
ans = min(ans, fromW[x][0]+fromY2[z][0])

# WtoZ => YtoX
edges2 = edges.copy()
nodes2 = [[] for i in range(N)]
for i in range(len(WtoZ)-1):
    edges2[(WtoZ[i], WtoZ[i+1])] = 0
for k, v in edges2.items():
    s, t = k
    c = v
    nodes2[s].append((t, c))
fromY2 = dijkstra(N, nodes2, y)
YtoX2 = get_route(y, x, fromY2)
common = set(WtoZ) & set(YtoX2)
if len(common):
    ans = min(ans, fromW[z][0]+fromY2[x][0])

# YtoX => WtoZ
edges2 = edges.copy()
nodes2 = [[] for i in range(N)]
for i in range(len(YtoX)-1):
    edges2[(YtoX[i], YtoX[i+1])] = 0
for k, v in edges2.items():
    s, t = k
    c = v
    nodes2[s].append((t, c))
fromW2 = dijkstra(N, nodes2, w)
WtoZ2 = get_route(w, z, fromW2)
common = set(YtoX) & set(WtoZ2)
if len(common):
    ans = min(ans, fromY[x][0]+fromW2[z][0])

# YtoZ => WtoX
edges2 = edges.copy()
nodes2 = [[] for i in range(N)]
for i in range(len(YtoZ)-1):
    edges2[(YtoZ[i], YtoZ[i+1])] = 0
for k, v in edges2.items():
    s, t = k
    c = v
    nodes2[s].append((t, c))
fromW2 = dijkstra(N, nodes2, w)
# WtoX2 = get_route(w, x, fromW2)
# common = set(YtoZ) & set(WtoX2)
# if len(common):
ans = min(ans, fromY[z][0]+fromW2[x][0])

print(ans)
