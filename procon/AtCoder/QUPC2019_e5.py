# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/qupc2018/editorial.pdf
　　　https://ikatakos.com/pot/programming_algorithm/contest_history/atcoder/2018/1020_qupc
・組み合わせ取らないでもできた。今回右から左に移動する値が、
　移動後に左にある個数*右にある個数 - 移動前に左にある個数*右にある個数
　が、変動する個数だ。
・あとnC2なら組み合わせ使わなくてもいけるんだな…。
・accumulate検証用(for回すより0.03秒くらいaccumulateのが速かった)
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product, accumulate
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

N = INT()
A = LIST()

# まず累積和取る
# sm = [0] + list(accumulate(A))
sm=[0]*(N+1)
for i in range(N):
    sm[i+1]+=sm[i]+A[i]

# 重複する数の集計
CR = Counter(sm)
# 現時点での区間和が0になる組み合わせ数
ans = 0
for k, v in CR.items():
    # nC2の組み合わせであればこれで出る
    ans += v * (v-1) // 2

CL = Counter()
mx = 0
cur = 0
# 区切り位置を1つずつ右にずらしていく
for i in range(N+1):
    v = sm[i]
    # この位置にすると1つ前の状態からいくつ変動するかを持つ
    cur += (CL[v]+1) * (CR[v]-1) - CL[v] * CR[v]
    CR[v] -= 1
    CL[v] += 1
    # 減らせる最大値を取る
    mx = max(mx, cur)
print(ans - mx)
