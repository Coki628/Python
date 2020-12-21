# -*- coding: utf-8 -*-

"""
・いもす
・区間の長さが100万あったからどうかなーと思ったけど、pythonでも0.6秒AC。
　やっぱaccumulateは結構速いのかね。
"""

import sys
from itertools import accumulate

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

N = INT()
M = 10 ** 6 + 1

imos = [0] * M
for i in range(N):
    s, t = MAP()
    imos[s] += 1
    imos[t] -= 1
imos = list(accumulate(imos))

cnt = 0
for i in range(1, M):
    # 1つ前が0で、次が0でない場所があればカウント
    if not imos[i-1] and imos[i]:
        cnt += 1
print(cnt)
