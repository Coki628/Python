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

def gcd(a, b):
    while b > 0:
        a, b = b, a%b
    return a
def lcm_base(x, y):
    return (x * y) // gcd(x, y)

N, M = MAP()
S = input()
T = input()

if N > M:
    if N % M == 0:
        gap = N // M
        # 割り切れる場合は間を確認する
        for i in range(M):
            # 決まってる文字が合わない
            if T[i] != S[gap*i]:
                print(-1)
                exit()
        print(N)
    else:
        lcm = lcm_base(N, M)
        N_gap = lcm // N
        M_gap = lcm // M
        for i in range(lcm):
            if i % N_gap == 0 and i % M_gap == 0:
                if S[i//N_gap] != T[i//M_gap]:
                    print(-1)
                    exit()
        print(lcm_base(N, M))
elif M > N:
    if M % N == 0:
        gap = M // N
        for i in range(N):
            if S[i] != T[gap*i]:
                print(-1)
                exit()
        print(M)
    else:
        lcm = lcm_base(N, M)
        N_gap = lcm // N
        M_gap = lcm // M
        for i in range(lcm):
            if i % N_gap == 0 and i % M_gap == 0:
                if S[i//N_gap] != T[i//M_gap]:
                    print(-1)
                    exit()
        print(lcm_base(N, M))
# N == M は完全一致のみ
else:
    if not all(S[i] == T[i] for i in range(N)):
            print(-1)
            exit()
    print(N)
