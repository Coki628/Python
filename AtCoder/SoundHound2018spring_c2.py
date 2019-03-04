# -*- coding: utf-8 -*-

"""
参考：https://naoyat.hatenablog.jp/entry/SoundHound2018Spring
　　　https://www.hamayanhamayan.com/entry/2018/01/28/172047
・参考1つ目に分かりやすく反例が出てる。
・最大独立集合問題
・無向グラフの中で、どの頂点間にも辺が無い最大成分を求める問題
・二部グラフでの最大独立集合の答えは「頂点数-最大マッチング数」
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product, accumulate
from operator import itemgetter, mul, xor
from copy import copy, deepcopy
from functools import reduce, partial
from fractions import Fraction
from string import ascii_lowercase, ascii_uppercase, digits

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def round(x): return int((x*2+1) // 2)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums): return reduce(lcm, nums, 1)
def gcd_list(nums): return reduce(gcd, nums, nums[0])
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

H,W=MAP()

# 四方に一回り大きいグリッドを作る
grid = list2d(H+2, W+2, '*')
for i in range(1,H+1):
    row = list(input())
    for j in range(1, W+1):
        grid[i][j] = row[j-1]

H2=H+2
W2=W+2
ans=0
bm=BipartiteMatching(H2*W2, H2*W2)
for i in range(1,H+1):
    for j in range(1, W+1):
        if grid[i][j]=='.':
            # 有効な頂点の数も一緒に数えておく
            ans+=1
            if grid[i][j-1]=='.':
                # i,jを1次元で表した値で管理する(ようは各頂点に一意な値が振れればいい)
                # 位置の偶奇で2部のどっちに入れるか決まる
                if (i+j)%2==0: bm.add(i*W2+j, i*W2+j-1)
                else: bm.add(i*W2+j-1, i*W2+j)
            if grid[i-1][j]=='.':
                if (i+j)%2==0: bm.add(i*W2+j, (i-1)*W2+j)
                else: bm.add((i-1)*W2+j, i*W2+j)
print(ans-bm.solve())
