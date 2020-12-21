# -*- coding: utf-8 -*-

"""
・速攻ではないけどほぼ悩まず進んで自力AC！
・500点でこれは嬉しい。
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
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

N, M, L = MAP()
T = LIST()

# 行きの依頼毎に、それに行った場合の往復期間を保持する
period = [[0] * 3 for i in range(M)]
for i in range(M):
    x, a = MAP()
    period[i][0] = x-1
    period[i][1] = a
# 帰りの依頼がある日を都市毎に保持する
back = [[] for i in range(N)]
for i in range(L):
    y, b = MAP()
    back[y-1].append(b)
for i in range(N):
    # 帰り依頼日をソートして末尾に番兵を仕込む
    back[i].sort()
    back[i].append(INF)

for i in range(M):
    city, start, _ = period[i]
    # ある依頼に行った場合の、最短の帰り依頼日のindex
    idx = bisect_right(back[city], start+T[city])
    # その帰り依頼で王都に戻って来れる日
    period[i][2] = back[city][idx] + T[city]
# 戻って来れる日が早い順にソート
period.sort(key=itemgetter(2))

ans = 0
cur = 0
for i in range(M):
    city, start, end = period[i]
    # 出発が間に合うなら貪欲にその依頼に行く
    if cur < start:
        cur = end
        ans += 2
        # 現在日がINFは最後の依頼から帰ってこないので復路分を引く
        if cur == INF:
            ans -= 1
print(ans)
