# -*- coding: utf-8 -*-

# 各種インポート
import sys, re
from collections import deque, defaultdict, Counter
from math import (
    sqrt, hypot, factorial, log10, log2,
    pi, sin, cos, acos, atan2, radians, degrees,
)
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product, accumulate
from operator import itemgetter, mul, add, xor
# from copy import copy, deepcopy
from functools import reduce, partial, lru_cache
from fractions import Fraction
from string import ascii_lowercase, ascii_uppercase, digits
from os.path import commonprefix

# 入出力高速化1(数値のみ)
# def input(): return sys.stdin.buffer.readline().strip()
# 入出力高速化2
# import io, os
# input = io.BytesIO(os.read(0,os.fstat(0).st_size)).readline
def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def round(x): return int((x*2+1) // 2)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(li): return reduce(lcm, li, 1)
def gcd_list(li): return reduce(gcd, li, 0)
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7
EPS = 10 ** -10

# 標準ライブラリのよりこっちのが速い(ただし2次元限定)
def deepcopy(li): return [x[:] for x in li]

# numpy系
import numpy as np
from scipy.sparse.csgraph import dijkstra, floyd_warshall

# 調査用
# import matplotlib.pyplot as plt 
# import pandas as pd

# 再帰呼び出しの回数制限(デフォルト1000)
sys.setrecursionlimit(10 ** 9)
# 再帰関数の前にこれ書くと速くなったりするらしい。
@lru_cache(maxsize=None)
def rec():
    rec()

# 小数点以下10桁まで表示(これやんないと自動でeとか使われる)
'{:.10f}'.format(3.1415)

# 文字列リバース
s = ''
s = s[::-1]

# int -> bin(str)
num = 1234
num = format(num, 'b')
# int -> bin(str) (8桁0埋め)
num = format(aN[i], '08b')
# bin(str) => int
num = int(num, 2)
# int -> 10進4桁0埋め(str)
num = format(num, '04')

# 二番目の要素でソート
aN = [[1, 2], [3, 1]]
aN.sort(key=lambda x: x[1])
# こっちのがちょっと速い
aN.sort(key=itemgetter(1))

# 四捨五入で整数に丸める
def round(x): return int((x*2+1) // 2)

# modの除算(フェルマーの小定理)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

# 配列要素全部掛け(総乗)
prod = partial(reduce, mul)
# これでもよさげ
def prod(nums): return reduce(mul, nums, 1)
prod([1, 2, 3])
np.prod([1, 2, 3])

# 余りの切り上げ(3つとも同じ)
# def ceil(a, b):
#     (a + b - 1) // b
#     (a - 1) // b + 1
#     return -(-a // b)

def LIS(A: list, equal=False) -> list:
    """ 最長増加部分列 """
    from operator import gt, ge
    from bisect import bisect_left, bisect_right

    # デフォルトは狭義のLIS(同値を含まない)
    compare = gt if not equal else ge
    bisect = bisect_left if not equal else bisect_right
    L = [A[0]]
    for a in A[1:]:
        if compare(a, L[-1]):
            # Lの末尾よりaが大きければ増加部分列を延長できる
            L.append(a)
        else:
            # そうでなければ、「aより小さい最大要素の次」をaにする
            # 該当位置は、二分探索で特定できる
            L[bisect(L, a)] = a
    return L

def FFT(A, B, L):
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
    return list(res)

def FFT(A, B):
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
    return list(res)

def bisearch_min(mn, mx, func):
    """ 条件を満たす最小値を見つける二分探索 """

    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            # 下を探しに行く
            ok = mid
        else:
            # 上を探しに行く
            ng = mid
    return ok

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索 """

    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

def bisearch_min(mn, mx, func, times):
    """ 条件を満たす最小値を見つける二分探索(小数用) """

    ok = mx
    ng = mn
    for _ in range(times):
        mid = (ok+ng) / 2
        if func(mid):
            # 下を探しに行く
            ok = mid
        else:
            # 上を探しに行く
            ng = mid
    return ok

def bisearch_max(mn, mx, func, times):
    """ 条件を満たす最大値を見つける二分探索(小数用) """

    ok = mn
    ng = mx
    for _ in range(times):
        mid = (ok+ng) / 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

def trisearch_min(lo, hi, func, ln=0):
    """ 三分探索 """

    m1 = l = lo
    m2 = r = hi
    while lo+2 < hi:
        m1 = (lo*2+hi) // 3
        m2 = (lo+hi*2) // 3
        res1 = func(m1)
        res2 = func(m2)
        if res1 <= res2:
            hi = m2
        else:
            lo = m1
    mn = INF
    ret = ()
    for i in range(max(m1-ln, l), min(m2+ln, r)):
        res = func(i)
        if res < mn:
            mn = res
            ret = (i, res)
    return ret

def trisearch_max(lo, hi, func, ln=0):
    """ 三分探索 """

    m1 = l = lo
    m2 = r = hi
    while lo+2 < hi:
        m1 = (lo*2+hi) // 3
        m2 = (lo+hi*2) // 3
        res1 = func(m1)
        res2 = func(m2)
        if res1 >= res2:
            hi = m2
        else:
            lo = m1
    mx = -INF
    ret = ()
    for i in range(max(m1-ln, l+1), min(m2+ln, r+1)):
        res = func(i)
        if res > mx:
            mx = res
            ret = (i, res)
    return ret

def trisearch_min(lo, hi, func, times):
    """ 三分探索(小数) """

    m1 = lo
    m2 = hi
    for _ in range(times):
        m1 = (lo*2+hi) / 3
        m2 = (lo+hi*2) / 3
        res1 = func(m1)
        res2 = func(m2)
        if res1 <= res2:
            hi = m2
        else:
            lo = m1
    return m1, m2

def trisearch_max(lo, hi, func, times):
    """ 三分探索(小数) """

    m1 = lo
    m2 = hi
    for _ in range(times):
        m1 = (lo*2+hi) / 3
        m2 = (lo+hi*2) / 3
        res1 = func(m1)
        res2 = func(m2)
        if res1 >= res2:
            hi = m2
        else:
            lo = m1
    return m1, m2

def bucket_sort(A, M, key):
    from itertools import accumulate
    """ バケットソート(タプルのソートが遅い時に使える) """

    N = len(A)
    C = [0] * (M+1)
    res = [()] * N
    for i in range(N):
        C[A[i][key]] += 1
    C = list(accumulate(C))

    for i in range(N-1, -1, -1):
        res[C[A[i][key]]-1] = A[i]
        C[A[i][key]] -= 1
    return res

def RLE(data):
    """ ランレングス圧縮 """
    from itertools import groupby

    return [(x, len(list(grp))) for x, grp in groupby(data)]

def compress(S):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(S)):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

def shakutori(N, K, A):
    """ 尺取法 """

    l = r = ans = 0
    sm = 0
    while l < N:
        # ここのand以下の条件は問題によって変わる
        while r < N and sm + A[r] <= K:
            sm += A[r]
            r += 1
        # ここで求める答えに足したりmax取ったりする
        ans += r - l
        # 左が右に追いついたら、右も左に合わせて+1する
        if l == r:
            # 左右同時に動くので、何も引く必要はない
            r += 1
        else:
            # 左を動かす分、合計から引く
            sm -= A[l]
        l += 1

def doubling(MAX, A):
    """ ダブリング """

    N = len(A)
    nxt = list2d(MAX, N, -1)
    for i, a in enumerate(A):
        nxt[0][i] = a
    for k in range(1, MAX):
        for i in range(N):
            nxt[k][i] = nxt[k-1][nxt[k-1][i]]
    return nxt

# 最大8ビット
def popcount(x):
    x -= (x >> 1) & 0x55
    x = (x & 0x33) + ((x >> 2) & 0x33)
    x = (x + (x >> 4)) & 0xf
    return x & 0xf

# 最大32ビット
def popcount(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0f0f0f0f
    i = i + (i >> 8)
    i = i + (i >> 16)
    return i & 0x3f

# 最大64ビット
def popcount(n):
    c = (n & 0x5555555555555555) + ((n>>1) & 0x5555555555555555)
    c = (c & 0x3333333333333333) + ((c>>2) & 0x3333333333333333)
    c = (c & 0x0f0f0f0f0f0f0f0f) + ((c>>4) & 0x0f0f0f0f0f0f0f0f)
    c = (c & 0x00ff00ff00ff00ff) + ((c>>8) & 0x00ff00ff00ff00ff)
    c = (c & 0x0000ffff0000ffff) + ((c>>16) & 0x0000ffff0000ffff)
    c = (c & 0x00000000ffffffff) + ((c>>32) & 0x00000000ffffffff)
    return c

# 最大128ビット
def popcount(n):
    c = (n & 0x55555555555555555555555555555555) + ((n>>1) & 0x55555555555555555555555555555555)
    c = (c & 0x33333333333333333333333333333333) + ((c>>2) & 0x33333333333333333333333333333333)
    c = (c & 0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f) + ((c>>4) & 0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f)
    c = (c & 0x00ff00ff00ff00ff00ff00ff00ff00ff) + ((c>>8) & 0x00ff00ff00ff00ff00ff00ff00ff00ff)
    c = (c & 0x0000ffff0000ffff0000ffff0000ffff) + ((c>>16) & 0x0000ffff0000ffff0000ffff0000ffff)
    c = (c & 0x00000000ffffffff00000000ffffffff) + ((c>>32) & 0x00000000ffffffff00000000ffffffff)
    c = (c & 0x0000000000000000ffffffffffffffff) + ((c>>64) & 0x0000000000000000ffffffffffffffff)
    return c

# 最大65536ビット
NN = 16
KK = (1 << (1 << NN)) - 1
I0 = KK // 3
I1 = KK // 5
I2 = KK // 17
I3 = KK // 257
def popcount(x):
    x -= (x >> 1) & I0
    x = (x & I1) + ((x >> 2) & I1)
    x = (x + (x >> 4)) & I2
    x = (x + (x >> 8)) & I3
    for k in range(4, 16):
        x += x >> 2**k
    return x & 0xffff

# 最大262144ビット
NN = 18
KK = (1 << (1 << NN)) - 1
I0 = KK // 3
I1 = KK // 5
I2 = KK // 17
I3 = KK // 257
I4 = KK // 65537
def popcount(x):
    x -= (x >> 1) & I0
    x = (x & I1) + ((x >> 2) & I1)
    x = (x + (x >> 4)) & I2
    x = (x + (x >> 8)) & I3
    x = (x + (x >> 16)) & I4
    for k in range(5, 18):
        x += x >> 2**k
    return x & 0x3ffff

# 無制限
def popcount(i):
    return bin(i).count('1')

def mat_pow(mat, init, K, MOD):
    """ 行列累乗 """

    def mat_dot(A, B, MOD):
        """ 行列の積 """

        # 1次元リストが来たら2次元の行列にする
        if not isinstance(A[0], list) and not isinstance(A[0], tuple):
            A = [A]
        if not isinstance(B[0], list) and not isinstance(A[0], tuple):
            B = [[b] for b in B]
        n1 = len(A)
        n2 = len(A[0])
        _ = len(B)
        m2 = len(B[0])
        res = list2d(n1, m2, 0)
        for i in range(n1):
            for j in range(m2):
                for k in range(n2):
                    res[i][j] += A[i][k] * B[k][j]
                    res[i][j] %= MOD
        return res

    def _mat_pow(mat, k, MOD):
        """ 行列matをk乗する """

        n = len(mat)
        res = list2d(n, n, 0)
        for i in range(n):
            res[i][i] = 1
        # 繰り返し二乗法
        while k > 0:
            if k & 1:
                res = mat_dot(res, mat, MOD)
            mat = mat_dot(mat, mat, MOD)
            k >>= 1
        return res

    # 行列累乗でK項先へ    
    res = _mat_pow(mat, K, MOD)
    # 最後に初期値と掛ける
    res = mat_dot(res, init, MOD)
    return [a[0] for a in res]

# 実数用
def gauss_jordan(A, b):
    """ ガウス・ジョルダン法(連立方程式の解) """

    N = len(A)
    B = list2d(N, N+1, 0)
    for i in range(N):
        for j in range(N):
            B[i][j] = A[i][j]
    # 行列Aの後ろにbを並べ同時に処理する
    for i in range(N):
        B[i][N] = b[i]
    
    for i in range(N):
        # 注目している変数の係数の絶対値が大きい式をi番目に持ってくる
        pivot = i
        for j in range(i, N):
            if abs(B[j][i]) > abs(B[pivot][i]):
                pivot = j
        B[i], B[pivot] = B[pivot], B[i]

        # 解がないか、一意でない
        if abs(B[i][i]) < EPS:
            return []

        # 注目している変数の係数を1にする
        for j in range(i+1, N+1):
            B[i][j] /= B[i][i]
        for j in range(N):
            if i != j:
                # j番目の式からi番目の変数を消去
                for k in range(i+1, N+1):
                    B[j][k] -= B[j][i] * B[i][k]
    
    res = [0] * N
    # 後ろに並べたbが解になる
    for i in range(N):
        res[i] = B[i][N]
    return res

# 2値用
def gauss_jordan(A, extended=False):
    """ ガウス・ジョルダン法(連立方程式の解) """

    N, M = len(A), len(A[0])
    res = [x[:] for x in A]
    rank = 0
    for col in range(M):
        if extended and col == M-1:
            break
        pivot = -1
        for row in range(rank, N):
            if res[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        res[rank], res[pivot] = res[pivot], res[rank]
        for row in range(N):
            if row != rank and res[row][col]:
                for i in range(M):
                    res[row][i] ^= res[rank][i]
        rank += 1
    for row in range(rank, N):
        if res[row][-1]:
            return (-1, [])
    return (rank, res)

# 2値用(ビット高速化版)
def gauss_jordan(A, extended=False):
    """ ガウス・ジョルダン法(連立方程式の解) """

    N = len(A)
    M = N + 1
    res = A[:]
    rank = 0
    for col in range(M):
        if extended and col == M-1:
            break
        pivot = -1
        for row in range(rank, N):
            if res[row]>>col & 1:
                pivot = row
                break
        if pivot == -1:
            continue
        res[rank], res[pivot] = res[pivot], res[rank]
        for row in range(N):
            if row != rank and res[row]>>col & 1:
                res[row] ^= res[rank]
        rank += 1
    # 解があるか確認
    for row in range(rank, N):
        if res[row]>>(M-1) & 1:
            return (-1, [])
    return (rank, res)
