# -*- coding: utf-8 -*-

"""
・行列累乗ライブラリ確認用
"""

import sys
from fractions import gcd

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

    def mat_pow(mat, k, MOD):
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
    res = mat_pow(mat, K, MOD)
    # 最後に初期値と掛ける
    res = mat_dot(res, init, MOD)
    return [a[0] for a in res]

A, B, M = MAP()
MOD = M

D = gcd(A, B)

# 必要な行列を構築
mat1 = list2d(2, 2, 0)
mat1[0][0] = 10
mat1[0][1] = 1
mat1[1][1] = 1
mat2 = list2d(2, 2, 0)
mat2[0][0] = pow(10, D, MOD)
mat2[0][1] = 1
mat2[1][1] = 1

# 行列累乗でA-1,B/D-1項先へ
res1 = mat_pow(mat1, [1, 1], A-1, MOD)
res2 = mat_pow(mat2, [1, 1], B//D-1, MOD)

# AとB/Dを合わせる
ans = res1[0] * res2[0] % MOD
print(ans)
