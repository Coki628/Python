"""
・結構さくっと自力AC！
・行列累乗
・状態の遷移を考えると、漸化式が作れる。(自分ノート参照)
・後はKがでかいので、普通にDPしないで行列累乗に落とし込めばOK。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

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

N, M, K, p, q = MAP()
A = LIST(N)

pq = fermat(p, q, MOD)
pq1 = (1 - pq) % MOD
mat = [
    [pq1, pq],
    [pq, pq1],
]
init = [sum(A[:M])%MOD, sum(A[M:])%MOD]

res = mat_pow(mat, init, K, MOD)
ans = res[0]
print(ans)
