# -*- coding: utf-8 -*-

"""
・こっちのライブラリだとWAだった。
"""

import sys
from collections import deque

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

	def __init__(self, n, m):
		self.n = n
		self.m = m
		# ペアとして有りうる相手を格納する
		self.E = [[] * (n+m+2) for i in range(n+m+2)]
		# 決まったペアを格納する
		self.match = [0] * (n+m+2)
		# 距離(付け替え候補を決める時の深さみたいなものが入ってる)
		self.dist = [0] * (n+m+2)
	
	def bfs(self):
		que = deque()
		for i in range(1, self.n+1):
			# マッチングの決まっていないノードをキューに詰める
			if not self.match[i]:
				self.dist[i] = 0
				que.append(i)
			else:
				self.dist[i] = float('inf')
		self.dist[0] = float('inf')
		while len(que) != 0:
			u = que.popleft()
			if u != 0:
				for v in self.E[u]:
					if self.dist[self.match[v]] == float('inf'):
						self.dist[self.match[v]] = self.dist[u] + 1
						# 付け替えの候補になりうるノードがここで追加されていく
						que.append(self.match[v])
		# フローを流せるエッジが存在しなければ、dist[0]がINFのまま(Falseを返して終了)
		return self.dist[0] != float('inf')

	def dfs(self, u):
		if u != 0:
			for v in self.E[u]:
				if self.dist[self.match[v]] == self.dist[u] + 1:
					if self.dfs(self.match[v]):
						# ペア確定(あとで変わることもある)
						self.match[v] = u
						self.match[u] = v
						return True
			# 今回決められなかったノード
			self.dist[u] = float('inf')
			return False
		return True
	
	def add(self, a, b):
		b += self.n
		self.E[a+1].append(b+1)
		self.E[b+1].append(a+1)

	def whois(self, x):
		return self.match[x+1]-1

	def solve(self):
		for i in range(self.n+self.m+1):
			self.match[i] = 0
		res = 0
		while self.bfs():
			for i in range(1, self.n+1):
				# 前から順に繋げる所は繋いでいって、
				# もっといいのがあれば後から付け替える
				if not self.match[i] and self.dfs(i):
					res += 1
		return res

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

bm=BipartiteMatching(N, M)
for i in range(N):
    for v in V[i][1]:
        bm.add(i, v)
bm.solve()

ans=0
for i in range(N):
    # ペアが決まっている宝の分を足していく
    if bm.whois(i)!=-1:
        ans+=V[i][0]
print(ans)
