# -*- coding: utf-8 -*-

"""
参考：https://naoyat.hatenablog.jp/entry/SoundHound2018Spring
　　　https://www.hamayanhamayan.com/entry/2018/01/28/172047
　　　https://ikatakos.com/pot/programming_algorithm/graph_theory/bipartite_matching
・参考1つ目に分かりやすく反例が出てる。
・最大独立集合問題
・無向グラフの中で、どの頂点間にも辺が無い最大成分を求める問題
・二部グラフでの最大独立集合の答えは「頂点数-最大マッチング数」
・ライブラリ動作確認
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
                # 位置の偶奇で二部のどっちに入れるか決まる
                if (i+j)%2==0: bm.add(i*W2+j, i*W2+j-1)
                else: bm.add(i*W2+j-1, i*W2+j)
            if grid[i-1][j]=='.':
                if (i+j)%2==0: bm.add(i*W2+j, (i-1)*W2+j)
                else: bm.add((i-1)*W2+j, i*W2+j)
print(ans-bm.solve())
