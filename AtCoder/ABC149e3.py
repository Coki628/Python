# -*- coding: utf-8 -*-

"""
参考：https://maspypy.com/atcoder-%e5%8f%82%e5%8a%a0%e6%84%9f%e6%83%b3-2019-12-29abc-149
・こっちの解法もやっておく。
・高速フーリエ変換(FFT)
・値の出現回数をカウントした2つの配列を渡すと、a+bを全組み合わせした時の各値の出現回数が得られる。
・Lは戻り値配列の長さで、max(A)+max(B)より大きい2冪を渡す。
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

def fft(A, B, L):
    """ 
    高速フーリエ変換(FFT)
        A：出現回数をカウントしたリスト
        B：出現回数をカウントしたリスト
        L：戻り値配列の長さ(max(A)+max(B)より大きい2冪を渡す)
    """
    import numpy as np
    from numpy.fft import rfft, irfft

    # FFT
    res = irfft(rfft(A, L) * rfft(B, L), L)
    # 四捨五入して整数に
    res = np.rint(res).astype(np.int64)
    return res

MAX = 10 ** 5
N, M = MAP()
A1 = [0] * (MAX+1)
A2 = [0] * (MAX+1)
for a in LIST():
    A1[a] += 1
    A2[a] += 1

L = 2 ** 18
res = fft(A1, A2, L)

ans = cnt = 0
for a in range(L-1, -1, -1):
    c = res[a]
    if cnt + c <= M:
        cnt += c
        ans += a * c
    else:
        # 同値でMを超える分の調整
        ans += a * (M - cnt)
        break
print(ans)
