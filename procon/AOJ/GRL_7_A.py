# -*- coding: utf-8 -*-

"""
・二部グラフの最大マッチング
"""

import sys
from collections import deque

sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
INF=float('inf')

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

N,M,E=MAP()
bm=BipartiteMatching(N, M)
for i in range(E):
    x,y=MAP()
    bm.add(x, y)

print(bm.solve())
