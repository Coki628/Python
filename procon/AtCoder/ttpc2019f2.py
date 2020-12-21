# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/ttpc2019/editorial.pdf
　　　https://www.hamayanhamayan.com/entry/2019/09/01/114634_3
・ダイクストラ応用
・2頂点が合流するまでのコストを合わせて、1つの新しい頂点とみなす。
・ゴール側は向き逆のグラフを使って同じことする。
・計算量は10万頂点20万辺のダイクストラ4発と40万辺に増やしてからもう1発。
　python2.3秒、pypy1.6秒でAC。
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

def dijkstra(N: int, nodes: list, src: int) -> list:
    """ ダイクストラ高速化版(頂点数, 隣接リスト(0-indexed), 始点) """
    from heapq import heappush, heappop

    # 頂点[ある始点からの最短距離] (経路自体を知りたい時はここに前の頂点も持たせる)
    res = [INF] * N
    # スタート位置
    que = [src]
    res[src] = 0
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
            if dist + cost < res[nxt]:
                res[nxt] = dist + cost
                # 距離*N+ノード番号 の形でキューに詰める
                heappush(que, (dist+cost)*N+nxt)
    # ノードsrcからの最短距離リストを返却
    return res

N, M = MAP()
w, x, y, z = MAP()
w -= 1
x -= 1
y -= 1
z -= 1
nodes = [[] for i in range(N)]
rev_nodes = [[] for i in range(N)]
for i in range(M):
    c, s, t = MAP()
    nodes[s-1].append((t-1, c))
    rev_nodes[t-1].append((s-1, c))

# w,yから各頂点
fromW = dijkstra(N, nodes, w)
fromY = dijkstra(N, nodes, y)
# 各頂点からx,z
toX = dijkstra(N, rev_nodes, x)
toZ = dijkstra(N, rev_nodes, z)

# 頂点wyを追加
nodes.append([])
for i in range(N):
    nodes[N].append((i, fromW[i]+fromY[i]))
# 頂点xzを追加
nodes.append([])
for i in range(N):
    nodes[i].append((N+1, toX[i]+toZ[i]))

fromWY = dijkstra(N+2, nodes, N)
# 共有部分がある時とない時で総コストの小さい方
ans = min(fromWY[N+1], fromW[x]+fromY[z])
if ans != INF:
    print(ans)
else:
    print('Impossible')
