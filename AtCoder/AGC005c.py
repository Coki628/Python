# -*- coding: utf-8 -*-

"""
・700点自力AC！
・絶対条件足りなそうと思いながらも、思いつくもの全て整理してまとめたら通った。
・N<=100だし絶対複雑なことすると思ったのに、これなら10万で良かったのでは。
"""

import sys
from collections import Counter

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
A = LIST()

C = sorted(Counter(A).items())

# 最短の頂点は1つか2つのみ、3つ以上はNG
if C[0][1] > 2:
    print('Impossible')
    exit()

# 最短の頂点が1つか2つか
is2src = C[0][1] == 2
# 最短の頂点の最大距離
src = C[0][0]
# 最短の頂点を根みたいに考えた時の木の深さ
depth = src - is2src + 1

# 最大距離の種類数は深さと一致しないとNG
if len(C) != depth:
    print('Impossible')
    exit()

for i, (k, v) in enumerate(C[1:], 1):
    prevk, prevv = C[i-1]
    # 最大距離が1ずつ進まないとNG
    if k != prevk + 1:
        print('Impossible')
        exit()
    # (最初以外は)各深さで頂点数が2つ以上ないとNG
    if 2 > v:
        print('Impossible')
        exit()
print('Possible')
