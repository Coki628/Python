# -*- coding: utf-8 -*-

"""
・これは愚直。解法出ず。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

P = INT()

def f(x):
    if x >= 10**16:
        return 0
    elif x <= 1:
        return 0
    elif x%2 == 0:
        return f(x//2) + 1
    else:
        return f(3*x+1) + 1

for i in range(100000):
    if f(i) == P:
        print(i)
        exit()
