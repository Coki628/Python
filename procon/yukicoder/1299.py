"""
・自力AC！
・期待値、行列累乗
・dp0[i] := i回終わった時の期待値
　dp1[i] := i回終わった時の全要素の平均
　みたいなDPを考えると、シンプルな1次元DPが作れる。
・制約がK<=10^18なので、普通にDPはできないけど、この遷移なら行列累乗に乗る。
・公式解見たら、sum取って計算一発みたいにできたらしい。。数学力さん。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 998244353
EPS = 10**-10

def div(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

def mat_pow(mat, init, K, MOD):
    """ 行列累乗 """

    def mat_dot(A, B, MOD):
        """ 行列の積 """

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
        while k > 0:
            if k & 1:
                res = mat_dot(res, mat, MOD)
            mat = mat_dot(mat, mat, MOD)
            k >>= 1
        return res

    res = _mat_pow(mat, K, MOD)
    res = mat_dot(res, init, MOD)
    return [a[0] for a in res]

N, K = MAP()
A = LIST()

# # dp0[i] := i回終わった時の期待値
# # dp1[i] := i回終わった時の全要素の平均
# dp0 = [0] * (K+1)
# dp1 = [0] * (K+1)
# dp0[0] = sum(A) % MOD
# dp1[0] = div(sum(A), N, MOD)
# for i in range(K):
#     dp0[i+1] = dp0[i] + dp1[i]*N
#     dp0[i+1] %= MOD
#     dp1[i+1] = (dp0[i] + dp1[i]*N) / N
#     dp1[i+1] %= MOD
# ans = dp0[K]
# print(ans)

invN = div(1, N, MOD)
mat = [
    [1, N],
    [invN, 1],
]
init = [sum(A) % MOD, div(sum(A), N, MOD)]

res = mat_pow(mat, init, K, MOD)
print(res[0])
