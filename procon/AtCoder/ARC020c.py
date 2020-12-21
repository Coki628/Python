# -*- coding: utf-8 -*-

"""
・さくっと自力AC！
・行列累乗
・ほぼARC050cでやったのと同じアプローチ。まあ、これ最近やったからね。
・規則的に繰り返す桁は、DP遷移から行列作って行列累乗に持ち込めば高速に処理できる。
・今回は途中に除算も使わないので、気にせずMOD取って進められる。
・ちなみに想定解はもっと色々難しい感じのことやってた。
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

def mat_dot(A, B):
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

def mat_pow(mat, n, k):
    """ n*nの行列matを行列累乗でk乗する """

    res = list2d(n, n, 0)
    for i in range(n):
        res[i][i] = 1
    # 繰り返し二乗法
    while k > 0:
        if k & 1:
            res = mat_dot(res, mat)
        mat = mat_dot(mat, mat)
        k >>= 1
    return res

# curを初期値として、aをl回繰り返した値を下の桁に繋げた値を返す
def calc(cur, a, l):
    # 必要な行列を構築
    mat = list2d(2, 2, 0)
    mat[0][0] = pow(10, len(str(a)), MOD)
    mat[0][1] = a
    mat[1][1] = 1
    # 行列累乗でL項先へ
    mat = mat_pow(mat, 2, l)
    # 最後に初期値と掛ける
    mat = mat_dot(mat, [cur, 1])
    return mat[0][0]

N = INT()
A = [0] * N
L = [0] * N
for i in range(N):
    A[i], L[i] = MAP()
MOD = INT()

cur = 0
for i in range(N):
    cur = calc(cur, A[i], L[i])
print(cur)
