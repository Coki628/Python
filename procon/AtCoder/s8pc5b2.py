# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/s8pc-5/editorial.pdf
・300点自力ACできなかった…。
・2点間距離/2は思ったけど、半径決まってる円をどう扱っていいか分からなかった。
・普通に全部min取ればいいだけなのか、どっか重なりそうな気がしちゃうけど、
　全ペア確認してるから大丈夫なのかな。
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

N,M=MAP()
circles=[]
for i in range(N):
    x,y,r=MAP()
    circles.append((x,y,r))
for i in range(M):
    x,y=MAP()
    circles.append((x,y,INF))

mn=INF
for i in range(len(circles)):
    for j in range(i+1, len(circles)):
        x1,y1,r1=circles[i]
        x2,y2,r2=circles[j]
        # 2点間の距離
        dist=hypot(x1-x2, y1-y2)
        # 両方半径が決まってなければ半々にする
        if r1==INF and r2==INF:
            mn=min(mn, dist/2)
        # 片方決まってる
        elif r1==INF:
            mn=min(mn, dist-r2, r2)
        elif r2==INF:
            mn=min(mn, dist-r1, r1)
        # 両方決まってる
        else:
            mn=min(mn, r1, r2)
print(mn)
