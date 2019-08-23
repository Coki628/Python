# -*- coding: utf-8 -*-

"""
・愚直TLE
・やっぱ組み合わせnCrはN^2くらいまでと思った方がいいな。。
・と思ったらpypy試しにやったら1.2秒で通った。N=32でもいけるんだな。
・解説見た。良く見たらKの最大6じゃないか。だからか。。
"""

import sys
from itertools import combinations

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
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, K = MAP()
A = LIST()

mn = INF
for comb in combinations(A, K):
    sm = sum(comb)
    cnt = 0
    for s in str(sm):
        s = int(s)
        if s < 5:
            cnt += s%5
        else:
            cnt += s%5 + 1
    mn = min(mn, cnt)
print(mn)
