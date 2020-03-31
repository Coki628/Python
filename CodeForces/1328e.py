# -*- coding: utf-8 -*-

"""
・終了後自力AC？(ただしツイートでLCA使える、は見てしまった。。)
・LCA(最小共通祖先)、ダブリング
・コンテスト中はDFSで何とか探せないか色々やってしまっている間に終了。まあ残り時間短かったしね。。
　LCA使えるって話聞いてその視点を考慮して考えたら、正解に辿り着けた。
・一番深い場所へのパスを基準にして、それと各頂点のLCAを比較に使うといい。
・根が1つに決まっている今回の設定なら、LCAを疑うもやるべきだったな。
　あのまま自分で考え続けてそこに行き着いたかは分からない。気付けるようになりたいね。
・計算量は20万頂点へのLCA構築とクエリで、pypyAC1.2秒。
"""

import sys
from collections import Counter

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def dfs(N, nodes, src):
    """ DFS(木、スタック、重みなし) """

    stack = [(src, -1, 0)]
    dist = [INF] * N
    while stack:
        u, prev, c = stack.pop()
        dist[u] = c
        nxt[0][u] = prev
        for v in nodes[u]:
            if v != prev:
                stack.append((v, u, c+1))
    return dist

N, Q = MAP()
nodes = [[] for i in range(N)]
for _ in range(N-1):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

MAX = 18
# nxt[k][v] := 頂点vから2^k遡った祖先
nxt = list2d(MAX, N, -1)
# dfsで各頂点の深さと親を取得
depths = dfs(N, nodes, 0)

# ダブリングのテーブル構築
for k in range(1, MAX):
    for v in range(N):
        if nxt[k-1][v] == -1: 
            continue
        nxt[k][v] = nxt[k-1][nxt[k-1][v]]

def get_lca(a, b):
    # 深い方をbにする
    if depths[a] > depths[b]:
        a, b = b, a
    # bをaと同じ高さまで持ってくる
    gap = depths[b] - depths[a]
    for i in range(MAX):
        # ビットの立っている所を辿れば、好きな高さに移動できる
        if gap & 1<<i:
            b = nxt[i][b]
    # この時点で一致すればそこがLCA
    if a == b:
        return a
    else:
        # aとbが一致する直前の高さまで持ってくる
        for i in range(MAX-1, -1, -1):
            a2 = nxt[i][a]
            b2 = nxt[i][b]
            if a2 != b2:
                a = a2
                b = b2
        # その親がLCA
        return nxt[0][a]

for _ in range(Q):
    li = [v-1 for v in LIST()[1:]]
    # 一番深い場所にある頂点uまでのパスを基準とする
    u = mx = -1
    for v in li:
        if depths[v] > mx:
            mx = depths[v]
            u = v
    # LCA(u, v)とvの距離が2以上離れる場所があればNG
    mx = -1
    for v in li:
        lca = get_lca(u, v)
        dist = depths[v] - depths[lca]
        mx = max(mx, dist)
    if mx <= 1:
        YES()
    else:
        NO()
