# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc061/editorial.pdf
・ベルマンフォード
・負の閉路があっても、そこに頂点Nが含まれなければ今回はOK
　って点に気付けなかったー。
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)

N, M = map(int, input().split())

edges = [None] * M
for i in range(M):
    a, b, c = map(int, input().split())
    # 0-indexed
    # コストの正負を逆転させれば最短経路問題として処理できる
    edges[i] = (a-1, b-1, -c)

def bellman_ford(N: int, edges: list, start: int) -> list:
    # 頂点[ある始点からの最短距離]
    # (経路自体を知りたい時はここに前の頂点も持たせる)
    verts = [float('inf')] * N
    verts[start] = 0
    # 各辺によるコストの置き換えを頂点数N-1回繰り返す
    for i in range(N-1):
        for src, dest, cost in edges:
            if verts[dest] > verts[src] + cost:
                verts[dest] = verts[src] + cost
    # 無限に減らせる場所がないか確認
    for i in range(N):
        for src, dest, cost in edges:
            if verts[dest] > verts[src] + cost:
                verts[dest] = verts[src] + cost
                if dest == N-1:
                    # 頂点Nがそれに該当していたら空リストを返却
                    return []
    # 問題なければ頂点リストを返却
    return verts

verts = bellman_ford(N, edges, 0)
if not verts:
    print('inf')
else:
    # 正負を戻して出力
    print(-verts[N-1])
