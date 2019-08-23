# -*- coding: utf-8 -*-

"""
・WA
"""

import sys

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
S = input()

buns = []
cnt = 0
# 各バンズの重なり具合を持っておく
for i in range(N):
    if S[i] == '(':
        cnt += 1
    elif S[i-1] == '(' and S[i] == ')':
        buns.append(cnt)
        cnt = 0

yummy = []
# 各所に肉を差し込んだ時の増加量をもっておく
for bun in buns:
    add = bun
    for i in range(bun*2-1):
        if i%2 == 1:
            add -= 1
        yummy.append(add)

# 大きい方から使う
yummy.sort(reverse=True)
print(sum(yummy[:K]))
