# -*- coding: utf-8 -*-

"""
・色々むずかった。実験したけど全く方針見えず。
"""

import sys
from fractions import gcd

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

N = INT()
A = [2, 3]

def check(x):
    sm = sum(A)
    for a in A:
        if gcd(a, sm-a+x) == 1:
            return False
    if gcd(x, sm) == 1:
        return False
    return True

while len(A) < N:
    i = 3
    while 1:
        if i not in A:
            if check(i):
                A.append(i)
                break
        i += 1
print(A)
