# -*- coding: utf-8 -*-

"""
参考：http://mayokoex.hatenablog.com/entry/2016/03/20/120833
・グラフ系、強連結成分分解、閉路の検出、DFS、ビット全探索
・割と近い方針で進んでたけど、制約2の扱いをどうグラフに表すかが分からなかった。
・結果的には、辺を張るor通らないの2択を全ての制約2に対して全通り試す、だった。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def SCC(N, edges):
    """ 強連結成分分解 """

    nodes1 = [[] for i in range(N)]
    nodes2 = [[] for i in range(N)]
    for u, v in edges:
        nodes1[u].append(v)
        nodes2[v].append(u)

    T = []
    visited = [False] * N
    def rec1(cur):
        visited[cur] = True
        for nxt in nodes1[cur]:
            if not visited[nxt]:
                rec1(nxt)
        # 行き止まったところから順にTに入れていく
        T.append(cur)

    # グラフが連結とは限らないので全頂点やる
    for u in range(N):
        if not visited[u]:
            rec1(u)

    visited = [False] * N
    group = [0] * N
    grpnum = 0
    def rec2(cur):
        group[cur] = grpnum
        visited[cur] = True
        for nxt in nodes2[cur]:
            if not visited[nxt]:
                rec2(nxt)

    # 逆順で進めるところまで行く
    for u in reversed(T):
        if not visited[u]:
            rec2(u)
            grpnum += 1
    return grpnum, group

N = INT()
M1 = INT()
edges = []
for i in range(M1):
    a, b = MAP()
    a -= 1; b -= 1
    edges.append((b, a))

# 制約1についての閉路確認
valid = [True] * N
_, group = SCC(N, edges)
C = Counter(group)
for i in range(N):
    if C[group[i]] >= 2:
        valid[i] = False

M2 = INT()
edges2 = []
for i in range(M2):
    a, b = MAP()
    a -= 1; b -= 1
    edges2.append((a, b))

def dfs(u, nodes, visited, valid):
    if not valid[u]:
        return False
    if visited[u]:
        return True
    visited[u] = 1
    for v in nodes[u]:
        # 1つでもvalidでない頂点に繋がるならここは通れない
        if not dfs(v, nodes, visited, valid):
            valid[u] = False
            return False
    return True

def check(edges, valid):
    # 閉路確認
    _, group = SCC(N, edges)
    C = Counter(group)
    for i in range(N):
        if C[group[i]] >= 2:
            valid[i] = False
    # DFSして到達不可になった頂点を数える
    nodes = [[] for i in range(N)]
    # 逆辺を張る
    for a, b in edges:
        nodes[b].append(a)
    visited = [0] * N
    for i in range(N):
        dfs(i, nodes, visited, valid)
    return sum(valid)

# 制約2について、どう扱うかの2択を全通り試す
ans = 0
for S in range(1<<M2):
    valid2 = valid[:]
    edges3 = []
    for i in range(M2):
        # 辺を張るor通らない
        if S & 1<<i:
            edges3.append(edges2[i])
        else:
            valid2[edges2[i][0]] = False
    ans = max(ans, check(edges + edges3, valid2))
print(ans)
