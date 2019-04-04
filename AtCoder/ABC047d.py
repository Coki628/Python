# -*- coding: utf-8 -*-

"""
・尺取法の変形
・自力AC
"""

import sys
import re
from collections import deque, Counter, defaultdict
from math import sqrt, hypot, factorial, pi, sin, cos, radians
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, product, combinations, combinations_with_replacement
from operator import itemgetter, mul
from copy import deepcopy
from functools import reduce, partial
from string import ascii_lowercase, ascii_uppercase, digits

def input(): return sys.stdin.readline().strip()
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, T = MAP()
aN = LIST()

mx = -INF
l, r = 0, 1
cnt = 0
# 外ループで左を動かす
while l < N:
    # 内ループは条件を満たす限り右を動かす
    while r < N and aN[l] < aN[r]:
        # 最高値更新ならその場所が操作対象になる
        if aN[r] - aN[l] > mx:
            mx = aN[r] - aN[l]
            cnt = 1
        # 同率最高値なら操作対象が1つ増える
        elif aN[r] - aN[l] == mx:
            cnt += 1
        mx = max(aN[r] - aN[l], mx)
        r += 1
    # 左端を右端に追いつかせる
    l = r
    r += 1
print(cnt)
