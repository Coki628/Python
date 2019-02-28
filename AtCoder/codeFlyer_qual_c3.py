# -*- coding: utf-8 -*-

"""
参考：https://betrue12.hateblo.jp/entry/2018/06/03/131350
　　　https://img.atcoder.jp/bitflyer2018-qual/editorial.pdf
・前計算は意外と合ってた。
・全体A-不要部分B
・Aは真ん中固定、Bは左固定で考える。
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product, accumulate
from operator import itemgetter, mul, xor
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

N,D=MAP()
X=LIST()

# ある都市の左側にある徒歩圏内の都市数
L=[0]*N
for i in range(1, N):
	idx=bisect_left(X, X[i]-D)
	L[i]=i-idx
# ある都市の右側にある徒歩圏内の都市数
R=[0]*N
for i in range(N-2, -1, -1):
	idx=bisect_right(X, X[i]+D)
	R[i]=idx-i-1

A=B=0
for i in range(N):
	# ある都市から徒歩圏内にある左右の都市の全組み合わせ
	A+=L[i]*R[i]
	# 各都市の右側(左でもいい)にある徒歩圏内の都市を2つ選ぶ組み合わせ
	# nC2=n*(n-1)//2
	B+=R[i]*(R[i]-1)//2
print(A-B)
