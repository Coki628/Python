# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/code-festival-2014-morning-easy/submissions/9301037
・自力ならず。。コーナーケース見抜けなかった。悔しいなー。
・2点からダイクストラやって、双方からの距離を比較。
・ただしグラフが非連結な場合もあるようで、到達不可 = 到達不可 も距離が一致した扱いで
　拾ってしまうので、それはOKケースから省かないといけない。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def dijkstra(nodes: list, src: int) -> list:
    """ ダイクストラ高速化版(隣接リスト(0-indexed), 始点) """
    from heapq import heappush, heappop

    N = len(nodes)
    # 頂点[ある始点からの最短距離]
    res = [INF] * N
    # スタート位置
    que = [src]
    res[src] = 0
    # キューが空になるまで
    while que:
        # 距離*N + 現在のノード
        cur = heappop(que)
        # 距離とノードに分ける
        dist = cur // N
        cur %= N
        if res[cur] < dist:
            continue
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
s, t = MAP()
s -= 1; t -= 1
nodes = [[] for i in range(N)]
for i in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    nodes[a].append((b, c))
    nodes[b].append((a, c))

# 始点・終点からダイクストラ
res1 = dijkstra(nodes, s)
res2 = dijkstra(nodes, t)

for i in range(N):
    # 両方から同じ最短距離で到達出来る場所があればOK
    # ※だたし両方から到達不可INFも一致してしまうので除く
    if res1[i] != INF and res1[i] == res2[i]:
        print(i + 1)
        exit()
print(-1)
