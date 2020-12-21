# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/mujin-pc-2018/editorial.pdf
　　　https://atcoder.jp/contests/mujin-pc-2018/submissions/4312123
・pypyでもTLEしたからC++でAC
・400万のループ2つ減らしたらpypyで通った。(2400万→1600万)
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

grid = list2d(N+2, M+2, '#')
for i in range(1, N+1):
    row = list(input())
    for j in range(1, M+1):
        grid[i][j] = row[j-1]

# 上下左右にいくつ進めるかを数えておく
# -1初期化することで、cntなしでうまく2方向同時に数えられる
left=list2d(N+2, M+2, -1)
up=list2d(N+2, M+2, -1)
for i in range(1,N+1):
    for j in range(1, M+1):
        if grid[i][j]=='.':
            left[i][j]=left[i][j-1]+1
            up[i][j]=up[i-1][j]+1
right=list2d(N+2, M+2, -1)
down=list2d(N+2, M+2, -1)
for i in range(N+1, 0, -1):
    for j in range(M+1, 0, -1):
        if grid[i][j]=='.':
            right[i][j]=right[i][j+1]+1
            down[i][j]=down[i+1][j]+1

ans=0
for i in range(1, N+1):
    for j in range(1, M+1):
        if grid[i][j]=='.':
            ans+=(up[i][j]+down[i][j])*(left[i][j]+right[i][j])
print(ans)
