# -*- coding: utf-8 -*-

"""
参考：https://ikatakos.com/pot/programming_algorithm/graph_theory/bipartite_matching
・二部最大マッチング
・ライブラリ動作確認
"""

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

N = int(input())
aN, bN, cN, dN = [0] * N, [0] * N, [0] * N, [0] * N
# 赤い点
for i in range(N):
	aN[i], bN[i] = map(int, input().split())
# 青い点
for i in range(N):
	cN[i], dN[i] = map(int, input().split())

# (赤い点の数, 青い点の数)
bm = BipartiteMatching(N, N)
for i in range(N):
	for j in range(N):
		if aN[i] < cN[j] and bN[i] < dN[j]:
			# ペア候補として適切な組を追加していく(0-indexed)
			bm.add(i, j)
print(bm.solve())
