# -*- coding: utf-8 -*-

"""
・300点の割に時間かかったけど何とか自力AC
・ビット全探索
・重複の確認にもビットを活用
・公式解は再帰でやってた。
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
from datetime import date

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

N,A,B,C=MAP()
L=[INT() for i in range(N)]
ABC=[A,B,C]

ABCs=[[] for i in range(3)]
for i, x in enumerate(ABC):
    # どの材料を使って作るとMPがいくらかかるかを全組み合わせ試す
    for j in range(1<<N):
        if j==0:
            ABCs[i].append((INF, j))
            continue
        sm=cnt=0
        for k in range(N):
            if j>>k&1:
                sm+=L[k]
                cnt+=1
        mp=(cnt-1)*10
        mp+=abs(x-sm)
        ABCs[i].append((mp, j))
    ABCs[i].sort()

mn=INF
# 3つだけなので、どれ優先するか順番全部試す
for perm in permutations([0,1,2]):
    msk=sm=0
    for i in perm:
        for j in range(len(ABCs[i])):
            mp,k=ABCs[i][j]
            # ビットに重複がなければ、その組み合わせは使える
            if msk|k==msk^k:
                sm+=mp
                msk+=k
                break
    mn=min(mn, sm)
print(mn)
