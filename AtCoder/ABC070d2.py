# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc070/editorial.pdf
・公式解が再帰だったのでやってみる。
・経路が木だと分かってると、こんなシンプルなDFSでいけるんだなー。
・ってpypyでTLEだったよこれ、ダイクストラより速そうなのに。
・pythonだとAC通った。しかもちょっとだけダイクストラより速く。分からん。
・こっちも入力速いやつにしたら0.7秒くらいになった。
　ちなみにpypyはそれでも1.3秒くらい…。
"""

import sys
def input(): return sys.stdin.readline().strip()

# 再帰呼び出しの回数制限(デフォルト1000)
sys.setrecursionlimit(10 ** 9)

N = int(input())
nodes = [[] for _ in range(N+1)]
for i in range(N-1):
    a, b, c = map(int, input().split())
    # nodes[出発ノード](到着ノード, コスト))
    nodes[a].append((b, c))
    # 無向グラフなのでこっちも入れる
    nodes[b].append((a, c))

memo = [0] * (N+1)
def dfs(cur, prev, dist):
    memo[cur] = dist
    for nxt, cost in nodes[cur]:
        if nxt == prev:
            continue
        dfs(nxt, cur, dist+cost)

Q, K = map(int, input().split())
dfs(K, -1, 0)

for i in range(Q):
    x, y = map(int, input().split())
    # 今回は全てKを経由するので、Kからの距離さえ分かれば全部出る
    print(memo[x] + memo[y])
