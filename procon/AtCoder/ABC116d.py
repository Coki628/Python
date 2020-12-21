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

N, K = MAP()
tdN = [[0]*2 for i in range(N)]
for i in range(N):
    t, d = MAP()
    tdN[i][0] = t
    tdN[i][1] = d

tdN.sort(key=itemgetter(1),reverse=True)
base = 0
bonus = 0
c = Counter()
items = defaultdict(list)
mx = defaultdict(int)
mn_idx = K-1
changes = []
for i in range(K):
    t, d = tdN[i]
    base += d
    if c[t] == 0:
        bonus += 1
    c[t] += 1
    if c[t] >= 2:
        heappush(changes, (d, i))
    heappush(items[t], d)
    mx[t] = max(d, mx[t])
bonus *= bonus

for i in range(K, N):
    base_tmp = base
    bonus_tmp = bonus
    t, d = tdN[i]
    # 既に持ってるネタならスキップ
    if c[t] != 0:
        continue
    else:
        # 交換候補と比較をする
        _, idx = heappop(changes)
        t2, d2 = tdN[idx]
        base_tmp -= d2 - d
        bonus_tmp = (sqrt(bonus_tmp)+1) ** 2
        if base+bonus < base_tmp+bonus_tmp:
            base = base_tmp
            bonus = bonus_tmp
        else:
            heappush(changes, (d2, idx))
    c[t] += 1
    mx[t] = max(d, mx[t])
print(base + bonus)
