"""
・さくっと自力AC
・FFTでぶん殴った。
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
EPS = 10 ** -10

def FFT(A, B):
    """ 
    高速フーリエ変換(FFT)
    """
    import numpy as np
    from numpy.fft import rfft, irfft

    MAXA = max(A)
    MAXB = max(B)
    C1 = [0] * (MAXA+1)
    C2 = [0] * (MAXB+1)
    for a in A:
        C1[a] += 1
    for b in B:
        C2[b] += 1
    L = 1
    k = 0
    while L <= MAXA + MAXB:
        k += 1
        L = 2**k
    res = irfft(rfft(C1, L) * rfft(C2, L), L)
    res = np.rint(res).astype(np.int64)
    return res

N, X = MAP()
A = LIST()

res = FFT(A, A)
if X < len(res):
    print(res[X])
else:
    print(0)
