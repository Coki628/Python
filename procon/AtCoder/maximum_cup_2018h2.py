# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/maximum-cup-2018/editorial.pdf
・蟻本演習3-5-6
・二部マッチング、割当問題、ワーシャルフロイド、二分探索
・1つ1つは理解できるけど、要素盛り沢山で自力で導くの大変ー、って感じ。
・ワーシャルフロイドの300^3=2700万がボトルネックでpythonTLE,pypyAC0.6秒。
・scipy使ってpythonAC0.7秒。
"""

import sys

from scipy.sparse.csgraph import floyd_warshall

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

def bisearch_min(mn, mx, func):
    """ 条件を満たす最小値を見つける二分探索 """

    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            # 下を探しに行く
            ok = mid
        else:
            # 上を探しに行く
            ng = mid
    return ok

def check(x):
    # ワープなしでのコストがx超の頂点ペアを集める
    pairs = []
    for i in range(K-1):
        u, v = A[i], A[i+1]
        if wf[u][v] > x:
            pairs.append((u, v))
    
    L = len(pairs)
    bm = BipartiteMatching(L, Q)
    for i, (u, v) in enumerate(pairs):
        for j, b in enumerate(B):
            # uからノーコストでbに飛んでからvへ向かうとする
            if wf[b][v] <= x:
                # x以内にできる場合は辺を張る
                bm.add(i, j)

    # ワープが必要な全ての頂点ペアが魔法石とマッチングできればOK
    cnt = bm.solve()
    return cnt == L

N, M = MAP()
G = list2d(N, N, INF)
for _ in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    G[a][b] = c
    G[b][a] = c

wf = floyd_warshall(G)

K = INT()
A = [a-1 for a in LIST()]
Q = INT()
B = [b-1 for b in LIST()]

ans = bisearch_min(-1, INF, check)
print(ans)
