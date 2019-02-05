# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc117/editorial.pdf
　　　https://atcoder.jp/contests/abc117/submissions/4167397
・XOR系
・各ビット0の多い所は1にするっていう方針は合ってた。
・ちょっと条件間違って書いてたのと、
　K以下に限定する時上の桁から見てくってのが足りなかった。
・計算量はO(NlogK)で多分400万くらい。pythonで1.8秒、pypyで0,5秒くらい。
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
Kb = format(K, 'b')
mxlen = len(Kb)

c = Counter()
for i in range(N):
    b = format(aN[i], '0' + str(mxlen) + 'b')
    b = b[::-1]
    # Kと同じ桁までの範囲で、各ビット0の所を集計する
    for j in range(len(b)):
        if b[j] == '0':
            c[j] += 1

num = ''
for i in range(mxlen):
    # ビット0が全体の過半数の所は、立てた方が得をする
    if c[i] > N / 2:
        num = '1' + num
    else:
        num = '0' + num

X = 0
for i in range(len(num)):
    if num[i] == '1':
        # 上の桁から順に、K以下の範囲内で足せるだけ足す
        if X + 2 ** (len(num)-i-1) <= K:
            X += 2 ** (len(num)-i-1)

ans = 0
for i in range(N):
    ans += X ^ aN[i]
print(ans)
