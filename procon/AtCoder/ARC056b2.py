# -*- coding: utf-8 -*-

"""
参考：http://arc056.contest.atcoder.jp/data/arc/056/editorial.pdf
　　　https://www.hamayanhamayan.com/entry/2016/06/27/022810
・公式解
・ダイクストラ応用
・最短経路コストの代わりに、その経路で通る最小の頂点番号の最大値、を持たせる。
・条件とか変えるとこんな使い方できるなんて知らなかったー。
"""

import sys
from heapq import heappush, heappop

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def LIST2(n): return [INT() for i in range(n)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

def dijkstra(N: int, nodes: list, src: int) -> list:
    """ ダイクストラ応用(頂点数, 隣接リスト(0-indexed), 始点) """

    # 頂点[ある始点からの経路で通る最小の頂点番号の最大値]
    res = [0] * N
    # スタート位置
    que = [(-src, src)]
    res[src] = src
    # キューが空になるまで
    while len(que):
        # srcからの経路で通る最小の頂点番号の最大値, 現在のノード
        mn, cur = heappop(que)
        mn = -mn
        # 出発ノードcurの到着ノードでループ
        for nxt in nodes[cur]:
            # ここまでの最小と次ノードで小さい方
            mntmp = min(mn, nxt)
            # 今回の頂点番号の方が大きい時
            if mntmp > res[nxt]:
                res[nxt] = mntmp
                # 最小の頂点番号をキューの優先度として、大きい方から先に処理するようにする
                heappush(que, (-res[nxt], nxt))
    # ノードsrcからの最短距離リストを返却
    return res

N, M, S = MAP()
nodes = [[] for i in range(N)]
for i in range(M):
    u, v = MAP()
    nodes[u-1].append(v-1)
    nodes[v-1].append(u-1)

res = dijkstra(N, nodes, S-1)
for i in range(N):
    # 自分以上の番号しか通らないならOK
    if res[i] >= i:
        print(i+1)
