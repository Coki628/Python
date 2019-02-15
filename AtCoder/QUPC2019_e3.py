# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/qupc2018/editorial.pdf
・区切る位置全走査は毎回nCrやると死ぬと思ってやらなかった。
・nCrやるのはその時右から左に移った1つだけでいいのと、
　logの事前テーブル作っておくことで間に合うようにできた。
・でもみんなこんな色々やってない。組み合わせ取らないでも出来るのかも…。
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product
from operator import itemgetter, mul
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
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

def init_fact_log(MAX: int) -> list:
    fact_log = [0] * (MAX)
    for i in range(1, MAX):
        fact_log[i] = fact_log[i-1] + log10(i)
    return fact_log

def nCr(n, r):
    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    return round(pow(10, fact_log[n] - fact_log[r] - fact_log[n-r]))

N = INT()
A = LIST()

# 階乗の事前テーブルをlogの世界で作っておく
fact_log = init_fact_log(N+2)

# まず累積和取る
sm = [0] * (N+1)
for i in range(N):
    sm[i+1] = sm[i] + A[i]

# 重複する数の集計
CR = Counter(sm)
# 現時点での区間和が0になる組み合わせ数
ans = 0
combs_r = defaultdict(int)
for k, v in CR.items():
    if v >= 2:
        comb = nCr(v, 2)
        # 数値毎の組み合わせ数を保持しておく
        combs_r[k] = comb
        ans += comb

CL = Counter()
combs_l = defaultdict(int)
mn = ans
# 区切り位置を1つずつ右にずらしていく
for i in range(N+1):
    v = sm[i]
    # 今回の値が右から1減って左に1増える
    CR[v] -= 1
    CL[v] += 1
    # 更新された個数での組み合わせ数
    comb_r = nCr(CR[v], 2)
    comb_l = nCr(CL[v], 2)
    # 前の数を引いて今回の数を足す
    ans += comb_l + comb_r - combs_l[v] - combs_r[v]
    # 最小値を取る
    mn = min(mn, ans)
    # 保持する組み合わせ数を今回の値に更新
    combs_l[v] = comb_l
    combs_r[v] = comb_r
print(mn)
