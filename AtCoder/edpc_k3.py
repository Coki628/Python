# -*- coding: utf-8 -*-

"""
参考：https://kyopro-friends.hatenablog.com/entry/2019/01/12/231000
　　　https://www.hamayanhamayan.com/entry/2019/01/09/002200
・後退解析：負け状態を押し付けられるかどうか
・メモ化再帰
・自分の得点等、手番固有の情報がないので、手番はメモしなくていい
・pypyはだめで、pythonでギリギリ1.9秒のAC
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

# メモ[山の残数] = 今の手番がこの局面から勝てるか
memo = [None] * (K+1)

def dfs(k):
    # 既に見た局面ならメモの内容を返す
    if memo[k] is not None:
        return memo[k]
    for i in range(N):
        # 山の残数より多くは取れない
        if k-aN[i] < 0:
            break
        # 1つでも負けを押し付ける手があれば勝てる
        if not dfs(k-aN[i]):
            memo[k] = True
            return True
    # 負けを押し付ける手がなかったので自分が負ける
    memo[k] = False
    return False

if dfs(K):
    print('First')
else:
    print('Second')
