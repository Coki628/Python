# -*- coding: utf-8 -*-

"""
・グリッドの反対側に値入れる調整すげー時間かかったー。
・まあそれでもWAだけども。
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product, accumulate
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
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums): return reduce(lcm, nums, 1)
def gcd_list(nums): return reduce(gcd, nums, nums[0])
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

C = Counter(A)
C2 = Counter()
for k, v in C.items():
    C2[v%4] += 1
    if v % 4 == 1:
        center = k

grid = list2d(N, N, 0)
if N % 2 == 0:
    for k, v in C2.items():
        if k != 0 and v != 0:
            print('NO')
            exit()
    l = []
    for k, v in C.items():
        for _ in range(v//4):
            l.append(k)
    k = 0
    for i in range(N//2):
        for j in range(N//2):
            grid[i][j] = l[k]
            grid[N-1-i][N-1-j] = l[k]
            grid[i][N-1-j] = l[k]
            grid[N-1-i][j] = l[k]
            k += 1
    print('YES')
    for i in range(N):
        print(*grid[i])

else:
    for k, v in C2.items():
        if (k == 2 or k == 3) and v != 0:
            print('NO')
            exit()
        if k == 1 and v != 1:
            print('NO')
            exit()
    C[center] -= 1
    grid[N//2][N//2] = center
    l = []
    for k, v in C.items():
        for _ in range(v//4):
            l.append(k)
    k = 0
    for i in range(N//2):
        for j in range(ceil(N, 2)):
            grid[i][j] = l[k]
            grid[N-1-i][N-1-j] = l[k]
            grid[i][N-1-j] = l[k]
            grid[N-1-i][j] = l[k]
            k += 1
    for i in range(N//2):
        grid[N//2][i] = grid[i][N//2]
        grid[N//2][N-1-i] = grid[N-1-i][N//2]
    print('YES')
    for i in range(N):
        print(*grid[i])
