# -*- coding: utf-8 -*-

"""
・調査用
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
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

import matplotlib.pyplot as plt

def f(b, n):
    if n < b:
        return n
    else:
        return f(b, n // b) + n % b

n = INT()
s = INT()

l = []
for i in range(100000):
    if i < 2: 
        l.append(0)
    else:
        l.append(f(i, n))

plt.plot(l)
plt.plot(s, 's', label='target s')
plt.ylabel('s')
plt.xlabel('b')
plt.show()
