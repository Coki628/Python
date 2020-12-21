# -*- coding: utf-8 -*-

"""
・文字列操作系
・ちょっとかかったけど自力AC！
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
def ceil(a, b=1): return int(-(-a // b))
def round(x): return int((x*2+1) // 2)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums): return reduce(lcm, nums, initial=1)
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

S = input()
N = len(S)

c = Counter(S)
if N < 26:
    for s in ascii_lowercase:
        if s not in c:
            print(S + s)
            exit()
else:
    # プレフィックスを1つずつ短くして試していく
    for i in range(N-1, -1, -1):
        pref = S[:i]
        S2 = S[i:]
        # 降順並び替え
        S3 = ''.join(sorted(S2, reverse=True))
        # 動かしたい場所が降順に並んでいたらもう動かせない
        if S2 != S3:
            # 交換対象の中で、置き換える文字より1つ大きいもの
            print(pref + S3[S3.index(S2[0])-1])
            exit()
    print(-1)
