# -*- coding: utf-8 -*-

"""
・結構何やるか悩んだけど自力AC
・ダイクストラ(優先度付きキュー)は思った以上に速いんだな。
・計算量はちゃんと作れてれば、O((E+V)logV)=2NlogN=20万*5=約100万(+Qの10万)
・pythonでギリAC(1.5秒くらい)、でも今回なぜかpypyでも速くならなかった。
・入力速いやつにしたら、0.8秒くらいになった！こんなに変わる時もあるんだなー。
　ちなみにpypyも同じくらいになった。
"""

import sys
def input(): return sys.stdin.readline().strip()

from heapq import heappop, heappush

# 今回はダイクストラするので、データの持ち方は隣接リスト
N = int(input())
nodes = [[] for _ in range(N+1)]
for i in range(N-1):
    a, b, c = map(int, input().split())
    # nodes[出発ノード](到着ノード, コスト))
    nodes[a].append((b, c))
    # 無向グラフなのでこっちも入れる
    nodes[b].append((a, c))

def dijkstra(nodes, start):
    # 最短距離メモ
    memo = [float('inf')] * (len(nodes))
    # スタート位置
    que = [(0, start)]
    memo[start] = 0
    # キューが空になるまで
    while len(que) != 0:
        # startからの距離, 現在のノード
        dist, cur = heappop(que)
        # 出発ノードcurの到着ノードでループ
        for nxt, cost in nodes[cur]:
            # 今回の経路のが短い時
            if memo[cur] + cost < memo[nxt]:
                memo[nxt] = memo[cur] + cost
                # 現在の移動距離をキューの優先度として、早い方から先に処理するようにする
                heappush(que, (memo[nxt], nxt))
    # ノードstartからの最短距離リストを返却
    return memo

Q, K = map(int, input().split())
# 引数：(ノードの隣接リスト, 出発ノード)
memo = dijkstra(nodes, K)

for i in range(Q):
    x, y = map(int, input().split())
    # 今回は全てKを経由するので、Kからの距離さえ分かれば全部出る
    print(memo[x] + memo[y])
