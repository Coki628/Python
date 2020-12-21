# -*- coding: utf-8 -*-

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

s = INT()
N = 1000000

aN = [0] * (N+1)
aN[0] = s
c = Counter()
c[aN[0]] += 1
for i in range(1, N+1):
    n = aN[i-1]
    if n % 2 == 0:
        aN[i] = n // 2
    else:
        aN[i] = 3*n + 1
    if c[aN[i]] == 0:
        c[aN[i]] += 1
    else:
        print(i+1)
        exit()
