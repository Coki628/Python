# -*- coding: utf-8 -*-

"""
・木の最短経路をDFSで2回やる。
・ちょっと悩んだけど自力AC。
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)

N = int(input())
nodes = [[] for _ in range(N+1)]
for i in range(N-1):
    a, b = map(int, input().split())
    # nodes[出発ノード](到着ノード, コスト))
    nodes[a].append((b, 1))
    # 無向グラフなのでこっちも入れる
    nodes[b].append((a, 1))

fennec = [float('inf')] * (N+1)
snuke = [float('inf')] * (N+1)
# 木の最短経路を算出するDFS
def dfs(memo: list, cur: int, prev: int, dist: int) -> None:
    memo[cur] = dist
    for nxt, cost in nodes[cur]:
        if nxt == prev:
            continue
        dfs(memo, nxt, cur, dist+cost)

# 2人の始点から各マスへの距離を出す
dfs(fennec, 1, -1, 0)
dfs(snuke, N, -1, 0)

fn = sn = 0
for i in range(2, N):
    # 距離の近い方がそのマスを取れる(同じ距離なら先手が取る)
    if fennec[i] <= snuke[i]:
        fn += 1
    else:
        sn += 1
# 取れるマスの多い方(同点だと後手が勝つ)
if fn > sn:
    print('Fennec')
else:
    print('Snuke')
