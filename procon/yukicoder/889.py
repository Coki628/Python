# -*- coding: utf-8 -*-

"""
・自力AC
・素数判定、平方数・立方数判定、完全数、誤差対策(N小さいけど念のため)
"""

import sys

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
EPS = 10 ** -10

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

N = INT()

if is_prime(N):
    print('Sosu!')
elif N >= 2 and abs(N**(1/2) - round(N**(1/2))) < EPS:
    print('Heihosu!')
elif N >= 2 and abs(N**(1/3) - round(N**(1/3))) < EPS:
    print('Ripposu!')
elif N in [6, 28]:
    print('Kanzensu!')
else:
    print(N)
