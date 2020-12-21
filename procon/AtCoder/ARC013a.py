# -*- coding: utf-8 -*-

"""
・高さ固定じゃなかったのでこれはWA
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

N, M, L = MAP()
P, Q, R = MAP()

# 高さLRは固定なので先に決めておく
if R > L:
    # 高さが足りなければ0確定
    print(0)
    exit()
L2 = L // R

# 縦横は入れ替え可能なので両方作る
N2 = N // P
M2 = M // Q
N3 = N // Q
M3 = M // P

print(max(N2*M2*L2, N3*M3*L2))
