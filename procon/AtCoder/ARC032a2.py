# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc032
・1~Nの等差数列の和の素数判定
・実は2より先ずっと合成数(確かにやってみると割り易そうな数が続く)
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

if N == 2:
    print('WANWAN')
else:
    print('BOWWOW')
