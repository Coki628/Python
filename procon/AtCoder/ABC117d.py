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
from copy import deepcopy
from functools import reduce, partial
from fractions import Fraction
from string import ascii_lowercase, ascii_uppercase, digits

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
aN = LIST()
Kb = format(K, 'b')
mxlen = len(Kb)

c = Counter()
for i in range(N):
    b = format(aN[i], 'b')
    b = b.zfill(mxlen)
    b = b[::-1]
    for j in range(len(b)):
        if b[j] == '0':
            c[j] += 1

num = ''
for i in range(mxlen):
    if c[i] > mxlen / 2:
        num = '1' + num
    else:
        num = '0' + num

while int(num, 2) > K:
    num = list(num)
    for i in range(len(num)-1, -1, -1):
        if num[i] == '1':
            num[i] = '0'
            break
    num = ''.join(num)
num = int(num, 2)

ans = 0
for i in range(N):
    ans += num ^ aN[i]
print(ans)
