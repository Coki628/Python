# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc139/editorial.pdf
　　　https://www.youtube.com/watch?v=UWbGRhF3Ozw
・本来辺だった各試合を頂点と見る視点
・DAG、トポロジカルソート
・最長経路、閉路検出
・頂点数横着してN**2で取ってたらTLE、ちゃんとN*(N-1)//2にしたらpypy1.5秒AC(pythonはTLE)。
・トポソはなにげに50万頂点くらいあっても有効に使えることが分かった。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

def topological_sort(N: int, edges: list) -> list:
    """ トポロジカルソート(頂点数、辺集合(DAG, 0-indexed)) """

    # ここからトポロジカルソート準備
    incnts = [0] * N
    outnodes = [[] for i in range(N)]
    for i in range(len(edges)):
        # 流入するノード数
        incnts[edges[i][1]] += 1
        # 流出先ノードのリスト
        outnodes[edges[i][0]].append(edges[i][1])
    # 流入ノード数が0であるノードのセットS
    S = set()
    for i in range(N):
        if incnts[i] == 0:
            S.add(i)
            # 流入ノード0のノードは始点に出来るので、最長距離を1にしておく
            dist[i] = 1

    # ここからトポロジカルソート
    L = []
    # 暫定セットが空になるまでループ
    while S:
        # 暫定セットから結果リストへ1つ入れる
        L.append(S.pop())
        # 確定させたノードから流出するノードでループ
        for node in outnodes[L[-1]]:
            # 流入ノード数を1減らす
            incnts[node] -= 1
            # 流入ノードが0なら暫定セットへ
            if incnts[node] == 0:
                S.add(node)
    # 閉路があって正しくソート出来なかった場合
    if len(L) != N:
        return []
    # ソートされた頂点のリストを返却
    return L

N = INT()

# 各試合を一意な値とするためのマッピング
mapping = list2d(N, N, -1)
cnt = 0
for i in range(N):
    for j in range(i+1, N):
        mapping[i][j] = cnt
        cnt += 1

A = []
for i in range(N):
    tmp = []
    for a in LIST():
        a -= 1
        a, b = min(a, i), max(a, i)
        # 各試合に一意な値を振って頂点とする
        tmp.append(mapping[a][b])
    A.append(tmp)

# 試合数で作ったグラフの辺集合と隣接リスト
MAXV = N*(N-1)//2
edges = []
nodes = [[] for i in range(MAXV)]
for i in range(N):
    for j in range(1, N-1):
        edges.append((A[i][j-1], A[i][j]))
        nodes[A[i][j-1]].append(A[i][j])

# トポロジカルソート(この時、始点となる頂点は初期値の距離1にする、閉路の検出も行う)
dist = [-1] * MAXV
res = topological_sort(MAXV, edges)
if not len(res):
    print(-1)
    exit()

# 依存性に従って、最長パスを更新していく
for u in res:
    for v in nodes[u]:
        dist[v] = max(dist[v], dist[u]+1)

print(max(dist))
