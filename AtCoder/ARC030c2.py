# -*- coding: utf-8 -*-

"""
・自力AC！
・強連結成分分解、DAG、メモ化再帰
・強連結成分分解でDAGに帰着させてDPする。
・こういうのあるよって話では聞いてたけど、初めてやった。時間かかったけど、うまくいって嬉しい。
・辞書順最小は文字列の最小値で管理。
・トポソしないでメモ化再帰版も練習のためにやってみた。
・出発点の頂点から確定していく形にしたかったので、有向辺の向きを逆にしてからメモ化再帰。
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
        # global grpnum
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

N, M, K = MAP()
C = list(input().split())
edges = []
for i in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    edges.append((a, b))

# 強連結成分分解
L, group = SCC(N, edges)

# 同グループ内のアルファベットを、頂点の昇順 = 辞書順 となるように置き換え
cgroup = [[] for i in range(L)]
vgroup = [[] for i in range(L)]
for i in range(N):
    g = group[i]
    cgroup[g].append(C[i])
    vgroup[g].append(i)
for i in range(L):
    cgroup[i].sort()
for i in range(L):
    for j, v in enumerate(vgroup[i]):
        C[v] = cgroup[i][j]

# グループ内は頂点の昇順につないでおく
nodes = [[] for i in range(N)]
edges2 = []
for i in range(L):
    for j, v in enumerate(vgroup[i][1:], 1):
        u = vgroup[i][j-1]
        # 逆辺にする
        nodes[v].append(u)

# グループ間の辺はグループ内末尾の頂点から、グループ内先頭の頂点に向かって張る
for a, b in edges:
    ga, gb = group[a], group[b]
    if ga == gb:
        continue
    a = vgroup[ga][-1]
    b = vgroup[gb][0]
    # 逆辺にする
    nodes[b].append(a)

# メモ化再帰
INF = '{'
memo = list2d(N, K+1, None)
def rec(u, k):
    if memo[u][k] is not None:
        return memo[u][k]
    if k == K:
        return ''
    res = INF
    for v in nodes[u]:
        # 頂点vの文字を使わない
        res = min(res, rec(v, k))
        # 頂点vの文字を使う
        res = min(res, rec(v, k+1) + C[v])
    memo[u][k] = res
    return res

ans = INF
for i in range(N):
    ans = min(ans, rec(i, 0))
    ans = min(ans, rec(i, 1) + C[i])
if ans == INF:
    print(-1)
else:
    print(ans)
