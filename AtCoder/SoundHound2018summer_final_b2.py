# -*- coding: utf-8 -*-

"""
参考：https://ikatakos.com/pot/programming_algorithm/contest_history/atcoder/2018/0728_soundhound2018-summer-final
　　　https://img.atcoder.jp/soundhound2018-summer-final/editorial.pdf
　　　https://ferin-tech.hatenablog.com/entry/2018/07/28/201656
　　　http://baitop.hatenadiary.jp/entry/2018/07/29/204143
・2つのDPをやる。
・いやー、遷移むずいよ。
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

N,K=MAP()
B=[INT() for i in range(N)]

# 薬品iまででの、iを0にした上での最大値
dp1=[0]*(N+1)
# 薬品iまででの、iを0にしなかった上での最大値
dp2=[0]*(N+1)

for i in range(N):
    if i<K:
        # K未満の薬品だけを0にすることはできないので、dp1はここでは使わない
        dp2[i+1]=dp2[i]+B[i]
    else:
        # max(0にした区間を1つずらして再度操作を行う, 今回の区間に対しては初めて操作を行う)
        dp1[i+1]=max(dp1[i], dp2[i-K+1])
        # max(0にしない(dp1), 0にしない(dp2))
        dp2[i+1]=max(dp1[i]+B[i], dp2[i]+B[i])
# それぞれの候補から最大を取る
print(max(dp1[N], dp2[N]))
