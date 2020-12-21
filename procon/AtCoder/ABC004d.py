# -*- coding: utf-8 -*-

"""
・部分点想定なのでWA
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

R, G, B = MAP()

def count(a):
    res = 0
    add = 1
    for i in range(1, a):
        res += add
        # 2回毎に加算される数が1増えていく
        if i%2 == 0:
            add += 1
    return res

print(count(R)+count(G)+count(B))
