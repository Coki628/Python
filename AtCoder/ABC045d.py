# -*- coding: utf-8 -*-

"""
参考：http://arc061.contest.atcoder.jp/data/arc/061/editorial.pdf
・各点が影響を与える9か所の正方形をメモしておく。
・これは解法出なかった。
・累積和で数え切れるほど小さくないしー、とか考えた。
・これは制限時間3秒だけど、pythonでちょうど2.0秒くらい、pypyで1.0秒くらいだった。
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

H, W, N = MAP()
# その黒い点が含まれる正方形の左上座標を持っておく
xy = []
for i in range(N):
    a, b = MAP()
    a -= 1
    b -= 1
    for j in range(3):
        for k in range(3):
            # 3*3の正方形を作れない座標は除く
            if 0 <= a-j < H-2 and 0 <= b-k < W-2:
                xy.append((a-j, b-k))
# 各座標の出現数を集計
c = Counter(xy)
# 0以外はこの集計結果から集められる
ans = [0] * 10
for k, v in c.items():
    ans[v] += 1
# 黒い点が0個の正方形の数 = 候補となりうる全正方形の数 - 黒い点が1個でも出現した正方形の数
ans[0] = (H-2)*(W-2) - len(c)

for num in ans:
    print(num)
