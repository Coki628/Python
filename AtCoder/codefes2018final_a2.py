# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/code-festival-2018-final/editorial.pdf
　　　https://atcoder.jp/contests/code-festival-2018-final/submissions/4131934
　　　https://atcoder.jp/contests/code-festival-2018-final/submissions/3624948
　　　https://atcoder.jp/contests/code-festival-2018-final/submissions/3632456
・この方針は思いつかなかった。
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
from copy import copy, deepcopy
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

N, M = MAP()

# 各ノードで、出ているエッジを距離毎に集計
lengths = [Counter() for i in range(N)]
for i in range(M):
    a, b, l = MAP()
    # [始点ノード番号][あるノードまでの距離] = その個数
    lengths[a-1][l] += 1
    lengths[b-1][l] += 1

cnt = 0
for i in range(N):
    for k, v in lengths[i].items():
        # 1270の場合は[k]と[2540-k]が同じになって重複するので別処理
        if k == 1270:
            cnt += v * (v-1)
        else:
            # 通常は普通にかけ合わせる
            cnt += v * lengths[i][2540-k]
# 往復分数えているので/2する
print(cnt // 2)
