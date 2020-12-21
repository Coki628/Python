# -*- coding: utf-8 -*-

"""
・自力AC
・結構考えた。同じ場所に2ないと補填に使えないのを、最終的に気付けて良かった。
・とはいえ300点はもっとサクサク解きたい。
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

N=INT()
A=LIST()
B=LIST()

cnt1=cnt2=0
for i in range(N):
    if A[i]>B[i]:
        # Bのが少ない合計*2(この分Aのどっかに足さないといけない)
        cnt1+=(A[i]-B[i])*2
    elif A[i]<B[i]:
        # Aのが少ない合計(この分で上記cnt1を受けきれるかどうか)
        # (こっちの合計が多い分には、同じ所でひたすら操作すれば帳尻が合うのでOK)
        # (同じ場所に2ないと意味ないので、1余る部分は数えない)
        cnt2+=(B[i]-A[i])-(B[i]-A[i])%2
if cnt1<=cnt2:
    Yes()
else:
    No()
