# -*- coding: utf-8 -*-

"""
・自力AC
・座標圧縮、2次元いもす
・たかだか2,3枚の重なりのために座圧いもすだるー、って思いながらも
　他に確実な方法も浮かばなかったのでこれで通した。
　条件整えれば貪欲さくっと通るんだろうなー、ってやりながら思ってたけど、
　意外とその条件整えるのがややこしかったぽくてみんな苦しんでたみたい。
"""

import sys
from itertools import accumulate

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def compress(S):
    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(S)):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

XY = []
S = set()
for i in range(3):
    x1, y1, x2, y2 = MAP()
    XY.append((x1, y1, x2, y2))
    S.add(x1)
    S.add(y1)
    S.add(x2)
    S.add(y2)

zipped, _ = compress(S)
N = len(zipped)
for i in range(3):
    x1, y1, x2, y2 = XY[i]
    XY[i] = (zipped[x1], zipped[y1], zipped[x2], zipped[y2])

imos = list2d(N+1, N+1, 0)
for x1, y1, x2, y2 in XY[1:]:
    imos[x1][y1] += 1
    imos[x2][y1] -= 1
    imos[x1][y2] -= 1
    imos[x2][y2] += 1
for i in range(N):
    imos[i] = list(accumulate(imos[i]))
for j in range(N):
    for i in range(N-1):
        imos[i+1][j] += imos[i][j]

x1, y1, x2, y2 = XY[0]
for x in range(x1, x2):
    for y in range(y1, y2):
        if not imos[x][y]:
            YES()
            exit()
NO()
