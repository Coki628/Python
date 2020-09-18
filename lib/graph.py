# -*- coding: utf-8 -*-

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


def dfs(nodes, src):
    """ DFS(木、再帰、重みなし) """

    N = len(nodes)
    dist = [INF] * N
    def rec(u, prev):

        for v in nodes[u]:
            if v != prev:
                dist[v] = dist[u] + 1
                rec(v, u)
        return
    dist[src] = 0
    rec(src, -1)

def dfs(nodes, src):
    """ DFS(木、再帰、重みあり) """

    N = len(nodes)
    dist = [INF] * N
    def rec(u, prev):

        for v, c in nodes[u]:
            if v != prev:
                dist[v] = dist[u] + c
                rec(v, u)
        return
    dist[src] = 0
    rec(src, -1)

def dfs(nodes, src):
    """ DFS(木、スタック、重みなし) """

    N = len(nodes)
    stack = [(src, -1)]
    dist = [INF] * N
    dist[src] = 0
    while stack:
        u, prev = stack.pop()
        for v in nodes[u]:
            if v != prev:
                dist[v] = dist[u] + 1
                stack.append((v, u))
    return dist

def dfs(nodes, src):
    """ DFS(木、スタック、重みあり) """

    N = len(nodes)
    stack = [(src, -1)]
    dist = [INF] * N
    dist[src] = 0
    while stack:
        u, prev = stack.pop()
        for v, c in nodes[u]:
            if v != prev:
                dist[v] = dist[u] + c
                stack.append((v, u))
    return dist

def bfs(nodes, src):
    """ BFS(一般グラフ、重みなし) """
    from collections import deque

    N = len(nodes)
    que = deque([src])
    dist = [INF] * N
    dist[src] = 0
    while que:
        u = que.popleft()
        for v in nodes[u]:
            if dist[v] != INF:
                continue
            dist[v] = dist[u] + 1
            que.append(v)
    return dist

def bfs(src):
    """ BFS(整数) """
    from collections import deque

    que = deque([src])
    dist = {}
    dist[src] = 0
    while que:
        u = que.popleft()
        # ここで遷移先への演算
        v = u
        if v not in dist:
            dist[v] = dist[u] + 1
            que.append(v)
    return dist

def dijkstra(nodes: list, src: int) -> list:
    """ ダイクストラ(隣接リスト(0-indexed), 始点) """
    from heapq import heappush, heappop

    N = len(nodes)
    # 頂点[ある始点からの最短距離] (経路自体を知りたい時はここに前の頂点も持たせる)
    res = [INF] * N
    # スタート位置
    que = [(0, src)]
    res[src] = 0
    # キューが空になるまで
    while que:
        # srcからの距離, 現在のノード
        dist, cur = heappop(que)
        if res[cur] < dist:
            continue
        # 出発ノードcurの到着ノードでループ
        for nxt, cost in nodes[cur]:
            # 今回の経路のが短い時
            if dist + cost < res[nxt]:
                res[nxt] = dist + cost
                # 現在の移動距離をキューの優先度として、早い方から先に処理するようにする
                heappush(que, (dist+cost, nxt))
    # ノードsrcからの最短距離リストを返却
    return res

def dijkstra(nodes: list, src: int) -> list:
    """ ダイクストラ高速化版(隣接リスト(0-indexed), 始点) """
    from heapq import heappush, heappop

    N = len(nodes)
    # 頂点[ある始点からの最短距離]
    res = [INF] * N
    # スタート位置
    que = [src]
    res[src] = 0
    # キューが空になるまで
    while que:
        # 距離*N + 現在のノード
        cur = heappop(que)
        # 距離とノードに分ける
        dist = cur // N
        cur %= N
        if res[cur] < dist:
            continue
        # 出発ノードcurの到着ノードでループ
        for nxt, cost in nodes[cur]:
            # 今回の経路のが短い時
            if dist + cost < res[nxt]:
                res[nxt] = dist + cost
                # 距離*N+ノード番号 の形でキューに詰める
                heappush(que, (dist+cost)*N+nxt)
    # ノードsrcからの最短距離リストを返却
    return res

def dijkstra(nodes: list, src: int) -> list:
    """ ダイクストラ高速化版経路付き(隣接リスト(0-indexed), 始点) """
    from heapq import heappush, heappop

    N = len(nodes)
    # 頂点(ある始点からの最短距離, 前の頂点)
    res = [(INF, -1) for i in range(N)]
    # スタート位置
    que = [src]
    res[src] = (0, -1)
    # キューが空になるまで
    while que:
        # 距離*N + 現在のノード
        cur = heappop(que)
        # 距離とノードに分ける
        dist = cur // N
        cur %= N
        if res[cur][0] < dist:
            continue
        # 出発ノードcurの到着ノードでループ
        for nxt, cost in nodes[cur]:
            # 今回の経路のが短い時
            if dist + cost < res[nxt][0]:
                res[nxt] = (dist+cost, cur)
                # 距離*N+ノード番号 の形でキューに詰める
                heappush(que, (dist+cost)*N+nxt)
    # ノードsrcからの最短距離と経路のリストを返却
    return res

def get_route(s, t, res):
    """ s,t間の経路を取得 """
    prev = t
    StoT = [t]
    while prev != s:
        prev = res[prev][1]
        if prev == -1:
            return None
        StoT.append(prev)
    StoT = StoT[::-1]
    return StoT

def dijkstra(G, src):
    """ O(V^2)のダイクストラ(動作確認してない) """

    N = len(G)
    dist = [INF] * N
    used = [False] * N

    dist[src] = 0
    while 1:
        v = -1
        for u in range(N):
            if not used[u] and (v == -1 or dist[u] < dist[v]):
                v = u
        if v == -1:
            break
        used[v] = True
        for u in range(N):
            dist[u] = min(dist[u], dist[v] + G[v][u])
    return dist

def bellman_ford(N: int, edges: list, src: int) -> list:
    """ ベルマンフォード(頂点数, 辺集合(0-indexed), 始点) """

    # 頂点[ある始点からの最短距離] (経路自体を知りたい時はここに前の頂点も持たせる)
    res = [INF] * N
    res[src] = 0
    # 各辺によるコストの置き換えを頂点数N-1回繰り返す
    for _ in range(N-1):
        for u, v, cost in edges:
            # INFからは更新しない
            if res[u] == INF:
                continue
            if res[v] > res[u] + cost:
                res[v] = res[u] + cost
    tmp = res[:]
    # 負の閉路(いくらでもコストを減らせてしまう場所)がないかチェックする
    for _ in range(N-1):
        for u, v, cost in edges:
            if res[u] == INF:
                continue
            if res[v] > res[u] + cost:
                # 検出した負閉路はすぐに-INFにする
                res[v] = -INF
    for i in range(N):
        if tmp[i] != res[i]:
            res[i] = -INF
    return res

def warshall_floyd(graph: list) -> list:
    """ ワーシャルフロイド(頂点数, 隣接行列(0-indexed)) """
    from copy import deepcopy

    N = len(graph)
    res = deepcopy(graph)
    for i in range(N):
        # 始点 = 終点、は予め距離0にしておく
        res[i][i] = 0
    # 全頂点の最短距離
    for k in range(N):
        for i in range(N):
            for j in range(N):
                res[i][j] = min(res[i][j], res[i][k] + res[k][j])
    # 負の閉路(いくらでもコストを減らせてしまう場所)がないかチェックする
    for i in range(N):
        if res[i][i] < 0:
            return []
    return res

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

def low_link(nodes):
    """ Low-Link(関節点aps(set)と橋bridges(list)を列挙する) """

    N = len(nodes)
    visited = [False] * N
    prenum = [0] * N
    parent = [0] * N
    lowest = [0] * N
    bridges = []
    timer = 1

    def rec(cur, prev, timer):
        # curを訪問した直後の処理
        prenum[cur] = lowest[cur] = timer
        timer += 1
        visited[cur] = True
        for nxt in nodes[cur]:
            if not visited[nxt]:
                # curからvへ訪問する直前の処理
                parent[nxt] = cur
                timer = rec(nxt, cur, timer)
                # nxtの探索が終了した直後の処理
                lowest[cur] = min(lowest[cur], lowest[nxt])
                # より近い経路を含まないなら橋とする
                if lowest[nxt] == prenum[nxt]:
                    # 番号の小さい方から入れる
                    bridges.append((min(cur, nxt), max(cur, nxt)))
            elif nxt != prev:
                # cur -> nxt がback-edgeの場合の処理
                lowest[cur] = min(lowest[cur], prenum[nxt])
        return timer
    # 必要な各値の取得(非連結に対応するため全頂点から)
    for i in range(N):
        if not visited[i]:
            timer = rec(i, -1, timer)

    # 間接点
    aps = set()
    # ルートの子ノードの数
    np = 0
    for i in range(1, N):
        p = parent[i]
        if p == 0:
            np += 1
        # 条件2の確認
        elif prenum[p] <= lowest[i]:
            aps.add(p)
    # 条件1の確認
    if np > 1:
        aps.add(0)

    return aps, bridges

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


class UnionFind:
    """ Union-Find木 """

    def __init__(self, n):
        self.n = n
        self.par = [i for i in range(n)]
        self.rank = [0] * n
        self.size = [1] * n
        self.tree = [True] * n
        self.grpcnt = n

    def find(self, x):
        """ 根の検索(グループ番号の取得) """
        t = []
        while self.par[x] != x:
            t.append(x)
            x = self.par[x]
        for i in t:
            self.par[i] = x
        return self.par[x]

    def union(self, x, y):
        """ 併合 """
        x = self.find(x)
        y = self.find(y)

        if x == y:
            self.tree[x] = False
            return
        if not self.tree[x] or not self.tree[y]:
            self.tree[x] = self.tree[y] = False

        self.grpcnt -= 1
        if self.rank[x] < self.rank[y]:
            self.par[x] = y
            self.size[y] += self.size[x]
        else:
            self.par[y] = x
            self.size[x] += self.size[y]
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1

    def is_same(self, x, y):
        """ 同じ集合に属するか判定 """
        return self.find(x) == self.find(y)

    def get_size(self, x=None):
        if x is not None:
            """ あるノードの属する集合のノード数 """
            return self.size[self.find(x)]
        else:
            """ 集合の数 """
            return self.grpcnt
    
    def is_tree(self, x):
        """ 木かどうかの判定 """
        return self.tree[self.find(x)]


class WeightedUnionFind:
    """ 重み付きUnion-Find木 """

    def __init__(self, n):
        self.par = [i for i in range(n)]
        self.rank = [0] * n
        # 根への距離を管理
        self.weight = [0] * n

    def find(self, x):
        """ 検索 """
        if self.par[x] == x:
            return x
        else:
            y = self.find(self.par[x])
            # 親への重みを追加しながら根まで走査
            self.weight[x] += self.weight[self.par[x]]
            self.par[x] = y
            return y

    def union(self, x, y, w):
        """ 併合 """
        rx = self.find(x)
        ry = self.find(y)
        # xの木の高さ < yの木の高さ
        if self.rank[rx] < self.rank[ry]:
            self.par[rx] = ry
            self.weight[rx] = w - self.weight[x] + self.weight[y]
        # xの木の高さ ≧ yの木の高さ
        else:
            self.par[ry] = rx
            self.weight[ry] = - w - self.weight[y] + self.weight[x]
            # 木の高さが同じだった場合の処理
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1

    def same(self, x, y):
        """ 同じ集合に属するか """
        return self.find(x) == self.find(y)

    def diff(self, x, y):
        """ xからyへのコスト """
        return self.weight[x] - self.weight[y]


class BipartiteMatching:
    """
    XとYの二部グラフの最大マッチング X={0,1,2,...|X|-1} Y={0,1,2,...,|Y|-1}
    edges[x]: xとつながるYの頂点のset
    match1[x]: xとマッチングされたYの頂点
    match2[y]: yとマッチングされたXの頂点
    """

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.edges = [set() for _ in range(n)]
        self.match1 = [-1] * n
        self.match2 = [-1] * m
 
    def dfs(self, v, visited):
        """
        :param v: X側の未マッチングの頂点の1つ
        :param visited: 空のsetを渡す（外部からの呼び出し時）
        :return: 増大路が見つかればTrue
        """
        for u in self.edges[v]:
            if u in visited:
                continue
            visited.add(u)
            if self.match2[u] == -1 or self.dfs(self.match2[u], visited):
                self.match2[u] = v
                self.match1[v] = u
                return True
        return False
    
    def add(self, a, b):
        self.edges[a].add(b)

    def whois1(self, a):
        """ :param: グループ1の頂点 :return: ペアになるグループ2の頂点 """
        return self.match1[a]

    def whois2(self, a):
        """ :param: グループ2の頂点 :return: ペアになるグループ1の頂点 """
        return self.match2[a]

    def solve(self):
        # 増大路発見に成功したらTrue(=1)。合計することでマッチング数となる
        return sum(self.dfs(i, set()) for i in range(self.n))


class Dinic:
    """ 最大流(Dinic)：O(E*V^2) """

    INF = 10**18

    def __init__(self, n):
        self.n = n
        self.links = [[] for _ in range(n)]
        self.depth = None
        self.progress = None
        self.pos = []

    def copy(self):
        res = Dinic(self.n)
        res.links = [[a[:] for a in b] for b in self.links]
        res.pos = [a[:] for a in self.pos]
        return res

    def add_edge(self, _from, to, cap):
        self.pos.append((_from, len(self.links[_from])))
        self.links[_from].append([cap, to, len(self.links[to])])
        self.links[to].append([0, _from, len(self.links[_from]) - 1])

    def bfs(self, s):
        from collections import deque

        depth = [-1] * self.n
        depth[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            for cap, to, _ in self.links[v]:
                if cap > 0 and depth[to] < 0:
                    depth[to] = depth[v] + 1
                    q.append(to)
        self.depth = depth

    def dfs(self, v, t, flow):
        if v == t:
            return flow
        links_v = self.links[v]
        for i in range(self.progress[v], len(links_v)):
            self.progress[v] = i
            cap, to, rev = link = links_v[i]
            if cap == 0 or self.depth[v] >= self.depth[to]:
                continue
            d = self.dfs(to, t, min(flow, cap))
            if d == 0:
                continue
            link[0] -= d
            self.links[to][rev][0] += d
            return d
        return 0

    def flow(self, s, t):
        INF = Dinic.INF
        flow = 0
        while True:
            self.bfs(s)
            if self.depth[t] < 0:
                return flow
            self.progress = [0] * self.n
            current_flow = self.dfs(s, t, INF)
            while current_flow > 0:
                flow += current_flow
                current_flow = self.dfs(s, t, INF)

    def get_edge(self, i):
        e = self.links[self.pos[i][0]][self.pos[i][1]]
        re = self.links[e[1]][e[2]]
        # from, to, cap, flow
        return (self.pos[i][0], e[1], e[0]+re[0], re[0])

    def get_edges(self):
        M = len(self.pos)
        res = []
        for i in range(M):
            res.append(self.get_edge(i))
        return res


class MinCostFlow:
    """ 最小費用流(ダイクストラ版)：O(F*E*logV) """

    INF = 10**18

    def __init__(self, N):
        self.N = N
        self.G = [[] for i in range(N)]
        self.pos = []

    def copy(self):
        res = MinCostFlow(self.N)
        res.G = [[a[:] for a in b] for b in self.G]
        res.pos = [a[:] for a in self.pos]
        return res

    def add_edge(self, fr, to, cap, cost):
        self.pos.append((fr, len(self.G[fr])))
        self.G[fr].append([to, cap, cost, len(self.G[to])])
        self.G[to].append([fr, 0, -cost, len(self.G[fr])-1])

    def flow(self, s, t, f):
        from heapq import heappush, heappop

        N = self.N; G = self.G
        INF = MinCostFlow.INF

        res = 0
        H = [0] * N
        prv_v = [0] * N
        prv_e = [0] * N

        while f:
            dist = [INF] * N
            dist[s] = 0
            que = [(0, s)]

            while que:
                c, v = heappop(que)
                if dist[v] < c:
                    continue
                for i, (to, cap, cost, _) in enumerate(G[v]):
                    if cap > 0 and dist[to] > dist[v] + cost + H[v] - H[to]:
                        dist[to] = r = dist[v] + cost + H[v] - H[to]
                        prv_v[to] = v; prv_e[to] = i
                        heappush(que, (r, to))
            if dist[t] == INF:
                return INF

            for i in range(N):
                H[i] += dist[i]

            d = f; v = t
            while v != s:
                d = min(d, G[prv_v[v]][prv_e[v]][1])
                v = prv_v[v]
            f -= d
            res += d * H[t]
            v = t
            while v != s:
                e = G[prv_v[v]][prv_e[v]]
                e[1] -= d
                G[v][e[3]][1] += d
                v = prv_v[v]
        return res

    def get_edge(self, i):
        e = self.G[self.pos[i][0]][self.pos[i][1]]
        re = self.G[e[0]][e[3]]
        # from, to, cap, flow, cost
        return (self.pos[i][0], e[0], e[1]+re[1], re[1], e[2])

    def get_edges(self):
        M = len(self.pos)
        res = []
        for i in range(M):
            res.append(self.get_edge(i))
        return res


class MinCostFlow:
    """ 最小費用流(ダイクストラ版2)：O(F*V^2) """

    INF = 10**18

    def __init__(self, N):
        self.N = N
        self.G = [[] for i in range(N)]
        self.pos = []

    def copy(self):
        res = MinCostFlow(self.N)
        res.G = [[a[:] for a in b] for b in self.G]
        res.pos = [a[:] for a in self.pos]
        return res

    def add_edge(self, fr, to, cap, cost):
        self.pos.append((fr, len(self.G[fr])))
        self.G[fr].append([to, cap, cost, len(self.G[to])])
        self.G[to].append([fr, 0, -cost, len(self.G[fr])-1])

    def flow(self, s, t, f):

        N = self.N; G = self.G
        INF = MinCostFlow.INF

        res = 0
        H = [0] * N
        prv_v = [0] * N
        prv_e = [0] * N

        while f:
            dist = [INF] * N
            dist[s] = 0
            used = [False] * N

            while True:
                v = -1
                for u in range(N):
                    if not used[u] and (v == -1 or dist[u] < dist[v]):
                        v = u
                if v == -1:
                    break
                used[v] = True
                for i, (to, cap, cost, _) in enumerate(G[v]):
                    if cap > 0 and dist[to] > dist[v] + cost + H[v] - H[to]:
                        dist[to] = dist[v] + cost + H[v] - H[to]
                        prv_v[to] = v; prv_e[to] = i
            if dist[t] == INF:
                return INF

            for i in range(N):
                H[i] += dist[i]

            d = f; v = t
            while v != s:
                d = min(d, G[prv_v[v]][prv_e[v]][1])
                v = prv_v[v]
            f -= d
            res += d * H[t]
            v = t
            while v != s:
                e = G[prv_v[v]][prv_e[v]]
                e[1] -= d
                G[v][e[3]][1] += d
                v = prv_v[v]
        return res

    def get_edge(self, i):
        e = self.G[self.pos[i][0]][self.pos[i][1]]
        re = self.G[e[0]][e[3]]
        # from, to, cap, flow, cost
        return (self.pos[i][0], e[0], e[1]+re[1], re[1], e[2])

    def get_edges(self):
        M = len(self.pos)
        res = []
        for i in range(M):
            res.append(self.get_edge(i))
        return res


class MinCostFlow:
    """ 最小費用流(ベルマンフォード版、負コストに対応可)：O(F*E*V) """

    INF = 10**18

    def __init__(self, N):
        self.N = N
        self.G = [[] for i in range(N)]
        self.pos = []
    
    def copy(self):
        res = MinCostFlow(self.N)
        res.G = [[a[:] for a in b] for b in self.G]
        res.pos = [a[:] for a in self.pos]
        return res

    def add_edge(self, fr, to, cap, cost):
        self.pos.append((fr, len(self.G[fr])))
        self.G[fr].append([to, cap, cost, len(self.G[to])])
        self.G[to].append([fr, 0, -cost, len(self.G[fr])-1])

    def flow(self, s, t, f):

        N = self.N; G = self.G
        INF = MinCostFlow.INF

        res = 0
        prv_v = [0] * N
        prv_e = [0] * N

        while f:
            dist = [INF] * N
            dist[s] = 0
            update = True

            while update:
                update = False
                for v in range(N):
                    if dist[v] == INF:
                        continue
                    for i, (to, cap, cost, _) in enumerate(G[v]):
                        if cap > 0 and dist[to] > dist[v] + cost:
                            dist[to] = dist[v] + cost
                            prv_v[to] = v; prv_e[to] = i
                            update = True
            if dist[t] == INF:
                return INF

            d = f; v = t
            while v != s:
                d = min(d, G[prv_v[v]][prv_e[v]][1])
                v = prv_v[v]
            f -= d
            res += d * dist[t]
            v = t
            while v != s:
                e = G[prv_v[v]][prv_e[v]]
                e[1] -= d
                G[v][e[3]][1] += d
                v = prv_v[v]
        return res

    def get_edge(self, i):
        e = self.G[self.pos[i][0]][self.pos[i][1]]
        re = self.G[e[0]][e[3]]
        # from, to, cap, flow, cost
        return (self.pos[i][0], e[0], e[1]+re[1], re[1], e[2])

    def get_edges(self):
        M = len(self.pos)
        res = []
        for i in range(M):
            res.append(self.get_edge(i))
        return res


class LCA:
    """ LCA(最小共通祖先) """

    def __init__(self, nodes, root):
        self.N = len(nodes)
        nv = 1
        MAX = 0
        while nv < self.N:
            nv *= 2
            MAX += 1
        self.MAX = MAX
        # nxt[k][v] := 頂点vから2^k遡った祖先
        self.nxt = list2d(MAX, self.N, -1)
        # dfsで各頂点の深さと親を取得
        self.depths = self.dfs(self.N, nodes, root)
        # ダブリングのテーブル構築
        for k in range(1, MAX):
            for v in range(self.N):
                if self.nxt[k-1][v] == -1: 
                    continue
                self.nxt[k][v] = self.nxt[k-1][self.nxt[k-1][v]]

    def dfs(self, N, nodes, src):
        stack = [(src, -1)]
        dist = [0] * N
        while stack:
            u, prev = stack.pop()
            for v in nodes[u]:
                if v != prev:
                    self.nxt[0][v] = u
                    dist[v] = dist[u] + 1
                    stack.append((v, u))
        return dist

    def get_lca(self, a, b):
        # 深い方をbにする
        if self.depths[a] > self.depths[b]:
            a, b = b, a
        # bをaと同じ高さまで持ってくる
        gap = self.depths[b] - self.depths[a]
        for i in range(self.MAX):
            # ビットの立っている所を辿れば、好きな高さに移動できる
            if gap & 1<<i:
                b = self.nxt[i][b]
        # この時点で一致すればそこがLCA
        if a == b:
            return a
        else:
            # aとbが一致する直前の高さまで持ってくる
            for i in range(self.MAX-1, -1, -1):
                a2 = self.nxt[i][a]
                b2 = self.nxt[i][b]
                if a2 != b2:
                    a = a2
                    b = b2
            # その親がLCA
            return self.nxt[0][a]
