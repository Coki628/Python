# -*- coding: utf-8 -*-

"""
・木の最短経路をDFSで2回やる。
・これはWA、勝ち負けの判定方法がいけてなかった。
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

dfs(fennec, 1, -1, 0)
dfs(snuke, N, -1, 0)

group = [set() for i in range(3)]
zero_to_n = fennec[N]
for i in range(2, N):
    # ノードがfennecの後ろにある
    if fennec[i] < snuke[i] and snuke[i] > zero_to_n:
        group[0].add(i)
    # ノードがsnukeの後ろにある
    elif fennec[i] > snuke[i] and fennec[i] > zero_to_n:
        group[1].add(i)
    # ノードが2人の間にある
    else:
        group[2].add(i)

# 自分の後方にノードがたくさんある方
if len(group[0]) > len(group[1]):
    print('Fennec')
elif len(group[0]) < len(group[1]):
    print('Snuke')
else:
    # 同率なら2人の間にあるノードの偶奇
    if zero_to_n % 2 == 0:
        print('Snuke')
    else:
        print('Fennec')
