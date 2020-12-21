# -*- coding: utf-8 -*-

"""
・自力AC！
・強連結成分分解、トポロジカルソート、DAG、DP
・強連結成分分解でDAGに帰着させてDPする。
・こういうのあるよって話では聞いてたけど、初めてやった。時間かかったけど、うまくいって嬉しい。
・辞書順最小は文字列の最小値で管理。
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
        nodes[u].append(v)
        edges2.append((u, v))

# グループ間の辺はグループ内末尾の頂点から、グループ内先頭の頂点に向かって張る
for a, b in edges:
    ga, gb = group[a], group[b]
    if ga == gb:
        continue
    a = vgroup[ga][-1]
    b = vgroup[gb][0]
    nodes[a].append(b)
    edges2.append((a, b))

# トポソしてDP
ts = topological_sort(N, edges2)
INF = '{'
dp = list2d(N, K+1, INF)
for i in range(N):
    dp[i][0] = ''
    dp[i][1] = C[i]
for u in ts:
    for k in range(K):
        for v in nodes[u]:
            # 頂点vの文字を使わない
            dp[v][k] = min(dp[v][k], dp[u][k])
            # 頂点vの文字を使う
            dp[v][k+1] = min(dp[v][k+1], dp[u][k] + C[v])

ans = INF
for i in range(N):
    ans = min(ans, dp[i][K])
if ans == INF:
    print(-1)
else:
    print(ans)
