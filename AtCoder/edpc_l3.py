# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/dp/submissions/4122950
　　　https://www.hamayanhamayan.com/entry/2019/01/09/004223
・区間DP
・メモ化再帰
・ゲーム木の探索
・得点は引数に入れないし、メモにも入れない。mxとmnの戻り値で管理する。
・とりあえずサンプル通ったけどTLE。再帰の900万はきつい。
・C++でほぼ同じ実装作って0.06秒くらいでAC
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

N = INT()
aN = LIST()

# メモ[左端位置][右端位置] = この局面からの結果
memo = [[None] * (N+1) for i in range(N+1)]

def dfs(turn, l, r):
    # 既に見た局面ならメモの内容を返す
    if memo[l][r] != None:
        return memo[l][r]
    # 全部取り切ったので終了
    if l > r:
        return 0
    # 先手番
    if turn == 0:
        mx = -INF
        mx = max(mx, dfs(1, l+1, r) + aN[l])
        mx = max(mx, dfs(1, l, r-1) + aN[r])
        # ここより深い所を探し終わったので結果をメモする
        memo[l][r] = mx
        return mx
    # 後手番
    else:
        mn = INF
        mn = min(mn, dfs(0, l+1, r) - aN[l])
        mn = min(mn, dfs(0, l, r-1) - aN[r])
        memo[l][r] = mn
        return mn

print(dfs(0, 0, N-1))
