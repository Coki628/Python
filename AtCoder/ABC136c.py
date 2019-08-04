# -*- coding: utf-8 -*-

"""
・300点自力AC
・逆から見た方が都合いいってのに気付くまでもたついた。。
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

N=INT()
H=LIST()

for i in range(N-2, -1, -1):
    if H[i]>H[i+1]:
        H[i]-=1

for i in range(N-1):
    if H[i]>H[i+1]:
        No()
        exit()
Yes()
