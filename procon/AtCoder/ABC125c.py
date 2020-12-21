# -*- coding: utf-8 -*-

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd
from functools import reduce

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def gcd_list(nums): return reduce(gcd, nums, nums[0])
def ceil(x, y=1): return int(-(-x // y))
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

N=INT()
A=LIST()

a=A[0]
idx=1
for i in range(1, N):
    b=gcd(a, A[i])
    if b<a:
        a=b
        idx=i

B=A[:]
B.pop(idx)
ans1=gcd_list(B)
B=A[:]
B.pop(idx-1)
ans2=gcd_list(B)
print(max(ans1, ans2))
