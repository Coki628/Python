# -*- coding: utf-8 -*-

"""
・速攻自力AC
・まあ300点だしな…。
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

S = input()
N = len(S)

T = 'gp' * (N//2) + 'g' * (N%2)
cnt = 0
for i in range(N):
    if S[i] == 'g' and T[i] == 'p':
        cnt += 1
    elif S[i] == 'p' and T[i] == 'g':
        cnt -= 1
print(cnt)
