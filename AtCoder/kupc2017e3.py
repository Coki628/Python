# -*- coding: utf-8 -*-

"""
参考：https://ikatakos.com/pot/programming_algorithm/graph_theory/bipartite_matching
・二部最大マッチング
・重み付きだけど、片側(宝箱)にしか依らないので、ソートして普通の二部最大マッチングでいける。
・汎用的に使えるようにクラスにまとめた。
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

N,M=MAP()
V=LIST()
# [0]:宝の価値, [1]:その宝を開けれる鍵リスト
V=[[V[i], []] for i in range(N)]

for i in range(M):
    x,y=MAP()
    V[x-1][1].append(i)
    V[y-1][1].append(i)
# 価値が高い方から決められるようにソートしておく
V.sort(reverse=True)

# グラフ読み取り
bm=BipartiteMatching(N, M)
for i in range(N):
    for j in V[i][1]:
        bm.add(i, j)
bm.solve()

ans=0
for i in range(N):
	if bm.whois1(i)!=-1:
		# ペアが決まった宝の分の価値を足していく
		ans+=V[i][0]
print(ans)
