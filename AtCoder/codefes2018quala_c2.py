# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2018/09/23/153000
　　　https://img.atcoder.jp/code-festival-2018-quala/editorial.pdf
・自分で分かったのは、割った回数が使うだろうなーっていう所までだった。
・こういうDPちゃんと自力で組めるようになりたい。
・数え上げDP
・3次元(2次元+2つの条件)DP
・pythonTLEでpypyACの0.9秒
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

N, K = MAP()
A = LIST()

# Aの各要素を0になるまで割った回数
div_cnt = [0] * N
for i in range(N):
    a = A[i]
    while a > 0:
        a //= 2
        div_cnt[i] += 1

# dp[i][k][0] := 数列の最初の i 個について、k 回の操作を実施して、操作後に 0 がないような場合の数
# dp[i][k][1] := 数列の最初の i 個について、k 回の操作を実施して、操作後に 0 があるような場合の数
dp = list3d(N+1, 3100, 2, 0)
dp[0][0][0] = 1

for i in range(N):
    for j in range(3001):
        for k in range(div_cnt[i]):
            # k < div_cnt[i] の間は0にならないので 0 => 0 に送る
            dp[i+1][j+k][0] += dp[i][j][0]
            dp[i+1][j+k][0] %= MOD
        # k = div_cnt[i] の時ちょうど0になるので 0 => 1 に送る
        dp[i+1][j+div_cnt[i]][1] += dp[i][j][0]
        dp[i+1][j+div_cnt[i]][1] %= MOD
        # 元々0がある場合を 1 => 1 に送る
        for k in range(div_cnt[i]+1):
            dp[i+1][j+k][1] += dp[i][j][1]
            dp[i+1][j+k][1] %= MOD

res = 0
# Kが大きすぎると全て0になっているので何も足さない
if K <= 3000:
    # K回操作後に0がない場合の数
    res += dp[N][K][0]
    res %= MOD

# 操作がK回以内(Kがでかすぎる場合を考慮して3000とminしておく)で0がある場合の数
for k in range(min(3000, K)+1):
    res += dp[N][k][1]
    res %= MOD

print(res)
