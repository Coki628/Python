# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/bitflyer2018-final/editorial.pdf
　　　https://betrue12.hateblo.jp/entry/2018/06/30/235952
・括弧列系
・出現する括弧を累積和して折れ線グラフで管理すると見通しがいい。
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

S=input()
N=len(S)

sm=[0]*(N+1)
for i in range(N):
	if S[i]=='(':
		sm[i+1]=sm[i]+1
	else:
		sm[i+1]=sm[i]-1

C=Counter()
ans=0
for i in range(N):
	C[sm[i]]+=1
	if sm[i+1]>sm[i]:
		# 下から上がってくる時には、その高さの前の値をリセットする
		C[sm[i+1]]=0
	else:
		# 下がる時にはその高さのカウントを足す
		ans+=C[sm[i+1]]
print(ans)
