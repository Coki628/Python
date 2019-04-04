# -*- coding: utf-8 -*-

"""
・尺取法
・まあWAだけども。
"""

import sys
from collections import deque, Counter, defaultdict
from math import sqrt, hypot, factorial, pi, sin, cos, radians
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, product, combinations, combinations_with_replacement
from operator import itemgetter, mul
from copy import deepcopy
from functools import reduce, partial

def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

S = input()
N = len(S)

l,r = 0,1
mx = 0
# 外ループで左を動かす
while l < N:
    # 内ループは条件を満たす限り右を動かす
    while r < N and S[l] != S[r]:
        r += 1
    # 同じ文字までの間隔が最大の所を探す
    if r < N:
        mx = max(r - l - 1, mx)
    if l == r:
        # 左が右に追いついたら、右も左に合わせて+1する
        r += 1
    l += 1

if mx == 0:
    if N % 2 == 0:
        print('Second')
    else:
        print('First')
else:
    if mx % 2 == 0:
        print('First')
    else:
        print('Second')
