# -*- coding: utf-8 -*-

"""
参考：https://qiita.com/drken/items/03c7db44ccd27820ea0d
・メモ化再帰でもやってみる。
・トポソ→DPよりこっちのが全然速かった！
　　トポソ→DP：1.1秒くらい
　　メモ化再帰：0.6秒くらい
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product
from operator import itemgetter, mul
from copy import deepcopy
from functools import reduce, partial
from fractions import Fraction
from string import ascii_lowercase, ascii_uppercase, digits

def input(): return sys.stdin.readline().strip()
def ceil(a, b=1): return int(-(-a // b))
def round(x): return int((x*2+1) // 2)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums): return reduce(lcm, nums, 1)
def gcd_list(nums): return reduce(gcd, nums, nums[0])
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, M = MAP()
nodes = [[] for i in range(N)]
edges = [None] * M
for i in range(M):
    x, y = MAP()
    nodes[x-1].append(y-1)
    edges[i] = ((x-1, y-1))

memo = [-1] * N
# メモ化再帰
def dfs(cur):
    if memo[cur] != -1:
        return memo[cur]
    mx = 0
    for dest in nodes[cur]:
        mx = max(mx, dfs(dest)+1)
    memo[cur] = mx
    return mx

ans = 0
for i in range(N):
    # メモはグローバルに持っているので、
    # 全ノードについてdfsしても無駄に潜ることはない。
    ans = max(ans, dfs(i))
print(ans)
