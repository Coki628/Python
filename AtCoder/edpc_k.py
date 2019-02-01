# -*- coding: utf-8 -*-

"""
・ゲーム木の探索
・メモ化再帰
・多分内容は合ってるんだけど、計算量1000万の再帰がTLEしちゃう。
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

N, K = MAP()
aN = LIST()

# メモ[手番][山の残数] = この局面から勝てるか
memo = [[None] * (K+1) for i in range(2)]

def dfs(turn, k):
    # 山がもう足りないので決着
    if k < 0:
        if turn == 0:
            return True
        else:
            return False
    # 先手番
    if turn == 0:
        # 既に見た局面ならメモの内容を返す
        if memo[turn][k] != None:
            return memo[turn][k]
        mx = -1
        for i in range(N):
            mx = max(mx, dfs(turn+1, k-aN[i]))
            # この先はやっても同じ
            if k-aN[i] < 0:
                break
        # ここより深い所を探し終わったので結果をメモする
        memo[turn][k] = mx
        return mx
    # 後手番
    else:
        if memo[turn][k] != None:
            return memo[turn][k]
        mn = float('inf')
        for i in range(N):
            mn = min(mn, dfs(turn-1, k-aN[i]))
            if k-aN[i] < 0:
                break
        memo[turn][k] = mn
        return mn

if dfs(0, K):
    print('First')
else:
    print('Second')
