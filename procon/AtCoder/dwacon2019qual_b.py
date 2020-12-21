# -*- coding: utf-8 -*-

"""
・かなり時間かかったけど自力AC
・公式解説の説明が謎だったけど、似たような解き方してる人はいた。
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

N, K = MAP()
A = LIST()

L = [[] for i in range(42)]
for i in range(N):
    tmp = [0] * (N+1)
    for j in range(i, N):
        # 制約はN=1000なので、累積和の区間和全列挙は間に合う
        tmp[j+1] += A[j] + tmp[j]
        k = 0
        while tmp[j+1] >= 2 ** k:
            k += 1
        # 先頭ビット毎に数を区分けする
        L[k-1].append(format(tmp[j+1], str(k) + 'b'))

for nums in reversed(L):
    # 大きい方から順番に試して、先頭ビットが立っているのがK個以上あればそこを使う
    if len(nums) >= K:
        L2 = copy(nums)
        break

# 先頭ビットで絞り込んだ候補に対して、2番目以下のビットを見る再帰処理
def rec(cur):
    global L2
    # 末尾まで見たら終了
    if cur == len(L2[0]):
        return
    cnt = 0
    flag = False
    L3 = []
    for i in range(len(L2)):
        if L2[i][cur] == '1':
            cnt += 1
            L3.append(L2[i])
            # 該当ビットが立っているものがK個以上あるか
            if cnt == K:
                flag = True
    if flag:
        # L3に絞り込んでいたものを採用する
        L2 = copy(L3)
        rec(cur+1)
    else:
        # 数が足りなければそのまま
        rec(cur+1)
    return

rec(1)

# L2を最適に整えたので、この中からK個の論理積を取れば答え
ans = int(L2[0], 2)
for i in range(1, K):
    ans &= int(L2[i], 2)
print(ans)
