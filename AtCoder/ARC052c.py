# -*- coding: utf-8 -*-

"""
参考：https://tokumini.hatenablog.com/entry/2018/10/06/140443
　　　http://arc052.contest.atcoder.jp/data/arc/052/editorial.pdf
・ダイクストラ応用
・公式解説には必要なタイプBの辺を限定する方法について書かれていたけど、難しかった。
・結果的には、記事を参考に、コストよりも前にBの使用回数を再優先事項として持つことでうまくいった。
・それに合わせて、距離の記録をnxtがcurになってから行うようにした。
　(こうしないと、Bの回数が少ない場合でも、距離が長いとそもそもキューに入らない)
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

def dijkstra(N: int, nodes: list, src: int) -> list:
    """ ダイクストラ(頂点数, 隣接リスト(0-indexed), 始点) """
    from heapq import heappush, heappop

    res = [INF] * N
    que = [(0, 0, src)]
    res[src] = 0
    while len(que) != 0:
        # bの使用回数, srcからの距離, 現在のノード
        b, dist, cur = heappop(que)
        if dist > res[cur]:
            continue
        # 距離の記録はこの頂点がcurになってから行う(nxtの時にやらない)
        res[cur] = dist
        for nxt, _type in nodes[cur]:
            if _type == 0:
                cost = 1
                b2 = b
            else:
                cost = b + 1
                b2 = b + 1
            if dist + cost < res[nxt]:
                # res[nxt] = dist + cost
                # bの回数を最優先でキューに入れる
                heappush(que, (b2, dist+cost, nxt))
    # ノードsrcからの最短距離リストを返却
    return res

N, M = MAP()
nodes = [[] for i in range(N)]
for i in range(M):
    c, a, b = MAP()
    nodes[a].append((b, c))
    nodes[b].append((a, c))

res = dijkstra(N, nodes, 0)
[print(a) for a in res]
