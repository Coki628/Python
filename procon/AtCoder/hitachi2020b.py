# -*- coding: utf-8 -*-

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

def fft(A, B):
    """ 
    高速フーリエ変換(FFT)
    """
    import numpy as np
    from numpy.fft import rfft, irfft

    # 出現数カウント
    MAXA = max(A)
    MAXB = max(B)
    C1 = [0] * (MAXA+1)
    C2 = [0] * (MAXB+1)
    for a in A:
        C1[a] += 1
    for b in B:
        C2[b] += 1
    # max(A)+max(B)より大きい2冪
    L = 1
    k = 0
    while L <= MAXA + MAXB:
        k += 1
        L = 2**k
    # FFT
    res = irfft(rfft(C1, L) * rfft(C2, L), L)
    # 四捨五入して整数に
    res = np.rint(res).astype(np.int64)
    return res

N, M, L = MAP()
A = LIST()
B = LIST()

res = list(fft(A, B))

mn = INF
for i, a in enumerate(res):
    if a != 0:
        mn = i
        break

for _ in range(L):
    x, y, c = MAP()
    x -= 1; y -= 1
    mn = min(mn, A[x] + B[y] - c)
print(mn)
