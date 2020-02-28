# -*- coding: utf-8 -*-

"""
・順列全探索、素数判定
"""

import sys
from itertools import permutations

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def is_prime(num):
    """ 素数判定 """
    from math import sqrt

    if num < 2: 
        return False
    if num in [2, 3, 5]: 
        return True
    if num % 2 == 0 or num % 3 == 0 or num % 5 == 0:
        return False
    # 疑似素数(2でも3でも割り切れない数字)で次々に割っていく
    prime = 7
    step = 4
    num_sqrt = sqrt(num)
    while prime <= num_sqrt:
        if num % prime == 0:
            return False
        prime += step
        step = 6 - step
    return True

S = input()

se = sorted(set(S))
N = len(se)
# 文字の種類数が5以下でないなら即NG
if N > 5:
    print(-1)
    exit()

# 13579からN個選ぶ全順列を試す
for perm in permutations(['1', '3', '5', '7', '9'], N):
    # 文字と数字の紐付け
    D = {}
    for i, a in enumerate(perm):
        D[se[i]] = a
    num = ''
    for s in S:
        num += D[s]
    num = int(num)
    # 素数判定
    if is_prime(num):
        print(num)
        exit()
print(-1)
