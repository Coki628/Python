# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/65136
　　　http://www.mathlion.jp/article/ar073.html
・整数系
・つまるところ、やりたいのはnの階乗がbで何回割り切れるか
・でかすぎて直接やる訳にはいかないので、素数べきにして各素数の数を見る
・階乗の素因数の個数が必要
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
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
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

def factorize(num: int, d: dict=Counter()) -> dict:
    # 終点はルート切り捨て+1
    end = int(sqrt(num)) + 1
    for i in range(2, end+1):
        cnt = 0
        # 素因数分解：小さい方から割れるだけ割って素数をカウント
        while num % i == 0:
            num //= i
            d[i] += 1
        # 1まで分解したら終了
        if num == 1:
            break
    # 最後に残ったnumは素数(ただし1^1は1^0なので数に入れない)
    if num != 1:
        d[num] += 1
    return d

n, b = MAP()

# 普通の素数べき
b_fact = factorize(b)

# 階乗の素数べき(bの素数べきに含まれるものだけやる)
n_fact = {}
for p, _ in b_fact.items():
    k = 1
    cnt = 0
    while n >= p ** k:
        cnt += n // (p ** k)
        k += 1
    n_fact[p] = cnt

cnt = INF
for k, v in b_fact.items():
    # 両方の素数べきを比べて、bを作るのに必要な素数がnにいくつあるか確認していく
    cnt = min(cnt, n_fact[k] // v)
# 一番少なかった所がボトルネックになるので、それが答え
print(cnt)
