# -*- coding: utf-8 -*-

"""
・自力AC！
・LCA(最小共通祖先)、ダブリング
・LCA使うやつがあるってのはネタバレしてたけど、多分これはそれがなくても気付けたはず。。
・最初pypyでもTLEで、なんでかと思ったら前処理の再帰DFSがボトルネックだった。
・今回特に戻り値から得るものはないので、スタック版にささっと置き換えたらAC。
・計算量は15万のN*logと10万クエリ*logで、pythonTLE,pypyAC0.8秒。
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

N = INT()
P = [p-1 for p in LIST(N)]

nodes = [[] for i in range(N)]
root = -1
for i, p in enumerate(P):
    if p == -2:
        root = i
    else:
        nodes[i].append(p)
        nodes[p].append(i)

MAX = 18
# nxt[k][v] := 頂点vから2^k遡った祖先
nxt = list2d(MAX, N, -1)
# dfsで各頂点の深さと親を取得
depths = dfs(N, nodes, root)

# ダブリングのテーブル構築
for k in range(1, MAX):
    for v in range(N):
        if nxt[k-1][v] == -1: 
            continue
        nxt[k][v] = nxt[k-1][nxt[k-1][v]]

def check(a, b):
    # aのが浅い位置にいる場合は部下ではない
    if depths[a] <= depths[b]:
        return False
    # aをbと同じ高さまで持ってくる
    gap = depths[a] - depths[b]
    for i in range(MAX):
        # ビットの立っている所を辿れば、好きな高さに移動できる
        if gap>>i & 1:
            a = nxt[i][a]
    # この時点で一致すれば直系の関係にある
    if a == b:
        return True
    else:
        return False

for _ in range(INT()):
    a, b = MAP()
    a -= 1; b -= 1
    if check(a, b):
        Yes()
    else:
        No()
