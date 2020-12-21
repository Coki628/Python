# -*- coding: utf-8 -*-

"""
・なんとか自力AC。でも時間かかり過ぎた…。
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians
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
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

def gcd(a, b):
    while b > 0:
        a, b = b, a%b
    return a
def lcm_base(x, y):
    return (x * y) // gcd(x, y)

N, M = MAP()
S = input()
T = input()

lcm = lcm_base(N, M)
# 何文字毎に文字が指定されるか
N_gap = lcm // N
M_gap = lcm // M
# 1文字ずつ全て確認することはない
for i in range(0, lcm, lcm//gcd(N, M)):
    # 文字の一致を確認する
    if S[i//N_gap] != T[i//M_gap]:
        print(-1)
        exit()
print(lcm)
