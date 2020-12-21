# -*- coding: utf-8 -*-

"""
・下手な貪欲っぽいものはWAから抜け出せず。
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
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
S = input()

for i in range(1, N):
    if S[-1]=='G' and S[i]=='G':
        No()
        exit()

fixed = [True] * N
i = 2
while i < N:
    if S[i-2]=='R' and S[i-1]=='G' and S[i]=='B':
        fixed[i-2] = fixed[i-1] = fixed[i] = False
        i += 1
        isG = False
        while i < N and S[i]=='B' or i < N-1 and S[i]=='G' and not isG:
            if S[i] == 'G':
                isG = True
            else:
                isG = False
            fixed[i] = False
            i += 1
    else:
        i += 1

i = N - 1 
while i > 1:
    if S[i-2]=='R' and S[i-1]=='G' and S[i]=='B':
        fixed[i-2] = fixed[i-1] = fixed[i] = False
        i -= 1
        isG = False
        while i > 1 and S[i-2]=='R' or i > 2 and S[i-2]=='G' and not isG:
            if S[i-2] == 'G':
                isG = True
            else:
                isG = False
            fixed[i-2] = False
            i -= 1
    else:
        i -= 1

# for i in range(1, N-1):
#     if S[i] == 'G' and (not fixed[i-1] or S[i-1]=='R') and (not fixed[i+1] or S[i+1]=='B'):
#         fixed[i-1] = fixed[i] = fixed[i+1] = False

if not any(a for a in fixed):
    Yes()
else:
    No()
