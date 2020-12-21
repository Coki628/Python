# -*- coding: utf-8 -*-

"""
・500点自力AC！
・gcdがlogくらいでも、10^18を10万回やったらダメだろうと思って
　ダメ元でやったのに通った。
　→log10^18が約60くらいだったから、600万ならまあいけるか…。
"""

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd
from functools import reduce

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums): return reduce(lcm, nums, 1)
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N,Z=MAP()
A=LIST()

l=[0]*N
for i in range(N):
    l[i]=gcd(Z, A[i])
print(lcm_list(l))
