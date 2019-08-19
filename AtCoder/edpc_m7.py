# -*- coding: utf-8 -*-

"""
・数え上げDP
・2次元+遷移先が多い系
・numpy実装、これはTLEだけどKが大きくなければこれもいけるはず。
　愚直から、3重ループは2重になってる。
"""

import sys
import numpy as np

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
A = LIST()

# dp[この子までの範囲で][飴を配る総数] = 場合の数
dp = np.zeros((N+1, K+1), dtype=np.int64)
# 0番目まで(誰もいない)の範囲で0個配る場合の数は1通り
dp[0,0] = 1

for i, a in enumerate(A):
    for k in range(a+1):
        # k個配る場合をまとめて遷移
        dp[i+1,k:K+1] += dp[i,:K+1-k]
        dp[i+1,k:K+1] %= MOD

print(dp[N][K])
