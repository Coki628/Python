# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-5、ダイクストラ応用
・その時いる高さに応じて、最適に動いた場合のコストを足す。
・でもこれ部分和DPほぼ関係ないんじゃあ。。
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

    res = [INF] * N
    que = [src]
    res[src] = 0
    while que:
        cur = heappop(que)
        dist = cur // N
        cur %= N

        # 現在の高さを取得
        cur_h = max(0, X - dist)

        for nxt, cost in nodes[cur]:
            
            # そのまま飛んだ時着地する高さ
            dist_h = cur_h - cost
            # 高すぎたら、降りてから飛ぶ
            if dist_h > H[nxt]:
                cost += dist_h - H[nxt]
            # 低すぎたら、登ってから飛ぶ
            elif dist_h < 0:
                # 今の木の高さが足りるなら登れる
                if H[cur] >= cur_h + (-dist_h):
                    cost += -dist_h
                else:
                    continue

            if dist + cost < res[nxt]:
                res[nxt] = dist + cost
                heappush(que, (dist+cost)*N+nxt)
    return res

N, M, X = MAP()
H = LIST(N)
nodes = [[] for i in range(N)]
for i in range(M):
    a, b, c = MAP()
    nodes[a-1].append((b-1, c))
    nodes[b-1].append((a-1, c))

res = dijkstra(N, nodes, 0)
if res[N-1] != INF:
    cur_h = max(0, X - res[N-1])
    # 最短経路 + 最後の木を登る高さ
    print(res[N-1] + (H[N-1] - cur_h))
else:
    print(-1)
