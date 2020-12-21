# -*- coding: utf-8 -*-

"""
参考：https://qiita.com/kakira9618/items/dfd87d89f2c446bd7a78
　　　https://atcoder.jp/contests/code-thanks-festival-2018/submissions/3664343
・数え上げDP
・DPテーブルのループの内側で、次の数が元々何個あるかのループを回す。
・最終的な答えに足し込む場所がi方向にもj方向にも複数ある。
・pythonはTLE、pypyでAC1.3秒くらい
・計算量O(T^3)=300^3=2700万くらい
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

T = INT()
aT = LIST()

# dp[i][j] = iがj個ある時の場合の数
dp = list2d(T+1, 601, 0)
# 初期化：1が使える数だけ1通りずつある
for i in range(aT[0]+1):
    dp[1][i] = 1

for i in range(1, T):
    # 偶数の時しか送らない(奇数個だと1つの数にできない)
    for j in range(0, 601, 2):
        # 次の数が元々k個ある場合を0～使える最大まで試す
        for k in range(aT[i]+1):
            # 元々ある次の数 + 今回の数//2 に遷移する
            dp[i+1][k+j//2] = (dp[i+1][k+j//2] + dp[i][j]) % MOD

ans = 0
for i in range(T+1):
    # その数1つだけがある状態の場合の数
    ans = (ans + dp[i][1]) % MOD

i = 1
while 2 ** i <= 600:
    # 最後の数で、最終的に1つにできる場合の数(2べきの場所)
    ans = (ans + dp[T][2**i]) % MOD
    i += 1

print(ans)
