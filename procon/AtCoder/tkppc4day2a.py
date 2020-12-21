# -*- coding: utf-8 -*-

import sys
from collections import defaultdict

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

X, Y = MAP()
MAX = 10**5

memo = defaultdict(lambda: -1)
def rec(x, y, cur):
    if x > abs(MAX) or y > abs(MAX):
        return
    if memo[(x, y)] != -1:
        return
    memo[(x, y)] = cur
    rec(x-1, y+2, cur+1)
    rec(x+1, y+2, cur+1)
rec(0, 0, 0)

print(memo[(X, Y)])
