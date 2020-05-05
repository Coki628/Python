# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・RGBそれぞれ使える文字を数えて、掛け合わせる。
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

RGB = [[]] * 3
for i in range(3):
    RGB[i] = input().split(',')
    if RGB[i][0] == 'NONE':
        RGB[i] = []

ans = 1
for i in range(3):
    cnt = 16 - len(RGB[i])
    ans *= cnt**2
print(ans)
