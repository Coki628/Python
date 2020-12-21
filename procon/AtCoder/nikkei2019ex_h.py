# -*- coding: utf-8 -*-

"""
・全然WAでした。。
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

N = INT()

bitcnt = 0
# 8進なので、3ビット毎に立っているか数える
for i in range(0, 61, 3):
    if (N >> i) & 1:
        bitcnt += 1

if bitcnt%2 == 1:
    print('Win')
else:
    print('Lose')
