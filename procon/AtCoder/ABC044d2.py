# -*- coding: utf-8 -*-

"""
・自力はここまで。
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import ceil, sqrt, hypot, factorial, pi, sin, cos, radians
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product
from operator import itemgetter, mul
from copy import deepcopy
from functools import reduce, partial
from fractions import Fraction
from string import ascii_lowercase, ascii_uppercase, digits

def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

def f(b, n):
    if n < b:
        return n
    else:
        return f(b, n // b) + n % b

n = INT()
s = INT()

if s > n:
    print(-1)
elif s == n:
    print(n+1)
elif ceil(n/2) < s < n:
    print(-1)
else:
    for i in range(2, n+1):
        if f(i, n) == s:
            print(i)
            exit()
    print(-1)
