# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/agc023/editorial.pdf
・累積和っぽいなーとは思ったんだけど、同じ数の組み合わせでいいってのが出なかった。
・累積和しちゃえば部分列の和はB-Aなんだから、BとAが同じ場所数えればいいんだよね…。
・でもこれ200点かなぁ…。
・階乗事前テーブル作成版
　→盛大にTLE。ローカルじゃいけてたのになんでだろ。
　とりあえず数がでかくて回数が少ない時はやんない方がいいね。
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
def lcm_list(nums): return reduce(lcm, nums, initial=1)
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

def init_factorial(MAX: int) -> list:
    # 階乗テーブル
    factorial = [1] * (MAX)
    factorial[0] = factorial[1] = 1
    for i in range(2, MAX):
        factorial[i] = factorial[i-1] * i
    return factorial

def nCr(n, r):
    # 10C7 = 10C3
    r = min(r, n-r)
    # 分子の計算
    numerator = factorial[n]
    # 分母の計算
    denominator = factorial[r] * factorial[n-r]
    return numerator // denominator

factorial = init_factorial(100001)

N = INT()
aN = LIST()

sm = [0] * (N+1)
for i in range(N):
    sm[i+1] = sm[i] + aN[i]

c = Counter(sm)
ans = 0
for k, v in c.items():
    if v >= 2:
        ans += nCr(v, 2)
print(ans)
