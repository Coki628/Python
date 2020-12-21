# -*- coding: utf-8 -*-

"""
・単一始点最短経路(負のコストを含む)
・ベルマンフォード
"""

import sys

sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
INF=float('inf')

# ベルマンフォード(頂点数, 辺集合(0-indexed), 始点)
def bellman_ford(N: int, edges: list, src: int) -> list:
    # 頂点[ある始点からの最短距離]
    # (経路自体を知りたい時はここに前の頂点も持たせる)
    res = [INF] * N
    res[src] = 0
    # 各辺によるコストの置き換えを頂点数N-1回繰り返す
    for i in range(N-1):
        for src, dest, cost in edges:
            if res[dest] > res[src] + cost:
                res[dest] = res[src] + cost
    # 負の閉路(いくらでもコストを減らせてしまう場所)がないかチェックする
    for src, dest, cost in edges:
        if res[dest] > res[src] + cost:
            # あったら空リストを返却
            return []
    # 問題なければ頂点リストを返却
    return res

N,M,r=MAP()
edges=[None]*M
for i in range(M):
    s,t,d=MAP()
    edges[i]=(s, t, d)

ans=bellman_ford(N, edges, r)

if not len(ans):
    print('NEGATIVE CYCLE')
    exit()

for i in range(N):
    if ans[i]==INF:
        ans[i]='INF'
for i in range(N):
    print(ans[i])
