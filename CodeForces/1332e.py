# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/contest/1332/submission/74952588
・行列累乗
・公式はO(1)解法でやってたけど、こっちならDP遷移が作れればできるので、
　いくらか自分でも導きやすそう。
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
INF = 10 ** 18
MOD = 998244353

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

N, M, L, R = MAP()

NM = N * M
LR = R - L + 1
if L % 2 == 0:
    ev = ceil(LR, 2)
    od = LR // 2
else:
    ev = ceil(LR, 2)
    od = LR // 2

# # dp0[i] := iマス目まで見て、偶数マスの数が偶数個になる通り数
# dp0 = [0] * (NM+1)
# # dp1[i] := iマス目まで見て、偶数マスの数が奇数個になる通り数
# dp1 = [0] * (NM+1)
# dp0[0] = 1
# for i in range(NM):
#     dp0[i+1] = (dp0[i] * od + dp1[i] * ev) % MOD
#     dp1[i+1] = (dp0[i] * ev + dp1[i] * od) % MOD
# if NM % 2 == 0:
#     ans = dp0[NM]
# else:
#     ans = (dp0[NM] + dp1[NM]) % MOD
# print(ans)

# 必要な行列を構築
mat = list2d(2, 2, 0)
mat[0][0] = mat[1][1] = od
mat[0][1] = mat[1][0] = ev
# 行列累乗
res = mat_pow(mat, [1, 0], NM, MOD)
if NM % 2 == 0:
    ans = res[0]
else:
    ans = (res[0] + res[1]) % MOD
print(ans)
