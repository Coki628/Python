# -*- coding: utf-8 -*-

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product
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
def prod(nums): return reduce(mul, nums, 1)
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums): return reduce(lcm, nums, 1)
def gcd_list(nums): return reduce(gcd, nums, nums[0])
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

def eratosthenes_sieve(n):
    table = [0] * (n + 1)
    prime_list = []
    for i in range(2, n + 1):
        if table[i] == 0:
            prime_list.append(i)
            for j in range(i + i, n + 1, i):
                table[j] = 1
    return prime_list

# 階乗の素数べき
def f(d, num):
    for p in eratosthenes_sieve(num):
        k = 1
        cnt = 0
        while num >= p ** k:
            cnt += num // (p ** k)
            k += 1
        if cnt != 0:
            d[p] += cnt
    return d

N, M = MAP()
rM = LIST()

mid = N / 2
mn = INF
r = 0
for i in range(M):
    if mn > abs(mid - rM[i]):
        mn = abs(mid - rM[i])
        r = rM[i]

C1 = Counter()
C2 = Counter()
C2[2] += N

# 階乗の素数べき
C1 = f(C1, N)
C2 = f(C2, r)
C2 = f(C2, N-r)

for k, v in C2.items():
    if v >= C1[k]:
        C2[k] -= C1[k]
        C1[k] -= C1[k]
    else:
        C1[k] -= v
        C2[k] -= v

p = prod([pow(k, v) for k, v in C1.items()]) / prod([pow(k, v) for k, v in C2.items()])
x = 1
while p < pow(10, -x):
    x += 1
print(x)
