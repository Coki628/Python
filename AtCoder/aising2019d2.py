# -*- coding: utf-8 -*-

"""
・シミュレーションして値の変化の規則性を探した。
・境界の位置と値の変わり方をリストに持たせて、各クエリで二分探索。
・ちょっと時間かかったけど自力AC！
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product
from operator import itemgetter, mul
from copy import deepcopy
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

N, Q = MAP()
A = LIST()

# xがある程度以上大きいと、交互に全部取られる
sm = sum([A[i] for i in range(N-1, -1, -2)])
borders = [(INF, sm)]
j = N-3
# 半分～後ろから2番目までの、取られる可能性のある部分を1つずつ確認する
for i in range(N-2, N//2-1, -1):
    # A[i]をギリギリ取られないで済むライン
    border =  (A[j] + A[i]) // 2
    # そこで得られる価値は、A[i] - 代わりに取らされるはずだった分
    val = A[i] - A[j]
    sm += val
    # (取られないで済むライン, その場合の総和)のリスト
    borders.append((border, sm))
    # 代わりに取らされるはずだった分は2ずつ戻る
    j -= 2
borders.sort()

for _ in range(Q):
    x = INT()
    # 必要な情報が整っていれば、二分探索でlogNで各クエリこなせる
    idx = bisect_left(borders, (x, 0))
    print(borders[idx][1])
