# -*- coding: utf-8 -*-

"""
参考：https://ikatakos.com/pot/programming_algorithm/graph_theory/maximum_flow
・二部グラフ特化のでなく、汎用的な最大流用のDinicクラスでやった版
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

class Dinic:
    def __init__(self, n):
        self.n = n
        self.links = [[] for _ in range(n)]
        self.depth = None
        self.progress = None
 
    def add_link(self, _from, to, cap):
        self.links[_from].append([cap, to, len(self.links[to])])
        self.links[to].append([0, _from, len(self.links[_from]) - 1])
 
    def bfs(self, s):
        depth = [-1] * self.n
        depth[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            for cap, to, rev in self.links[v]:
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
 
    def max_flow(self, s, t):
        flow = 0
        while True:
            self.bfs(s)
            if self.depth[t] < 0:
                return flow
            self.progress = [0] * self.n
            current_flow = self.dfs(s, t, float('inf'))
            while current_flow > 0:
                flow += current_flow
                current_flow = self.dfs(s, t, float('inf'))

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
dinic=Dinic(H2*W2)
for i in range(1,H+1):
    for j in range(1, W+1):
        if grid[i][j]=='.':
            # 始点～(i,j)と(i,j)～終点を二部のどっちかによってつなぐ
            if (i+j)%2==0:
                dinic.add_link(0, i*W2+j, 1)
            else:
                dinic.add_link(i*W2+j, H2*W2-1, 1)
            # 有効な頂点の数も一緒に数えておく
            ans+=1
            if grid[i][j-1]=='.':
                # i,jを1次元で表した値で管理する(ようは各頂点に一意な値が振れればいい)
                # 位置の偶奇で二部のどっちに入れるか決まる
                if (i+j)%2==0: 
                    dinic.add_link(i*W2+j, i*W2+j-1, 1)
                else: 
                    dinic.add_link(i*W2+j-1, i*W2+j, 1)
            if grid[i-1][j]=='.':
                if (i+j)%2==0: 
                    dinic.add_link(i*W2+j, (i-1)*W2+j, 1)
                else: 
                    dinic.add_link((i-1)*W2+j, i*W2+j, 1)

print(ans-dinic.max_flow(0, H2*W2-1))
