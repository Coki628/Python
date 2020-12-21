"""
・さくっと自力AC
・連立方程式、ガウス・ジョルダン
・式変形考えるのがだるすぎて、ライブラリで殴った。。
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
MOD = 10**9 + 7
EPS = 10**-10

def gauss_jordan(A, b):
    """ ガウス・ジョルダン法(連立方程式の解) """

    N = len(A)
    B = list2d(N, N+1, 0)
    for i in range(N):
        for j in range(N):
            B[i][j] = A[i][j]
    for i in range(N):
        B[i][N] = b[i]
    
    for i in range(N):
        pivot = i
        for j in range(i, N):
            if abs(B[j][i]) > abs(B[pivot][i]):
                pivot = j
        B[i], B[pivot] = B[pivot], B[i]

        if abs(B[i][i]) < EPS:
            return []

        for j in range(i+1, N+1):
            B[i][j] /= B[i][i]
        for j in range(N):
            if i != j:
                for k in range(i+1, N+1):
                    B[j][k] -= B[j][i] * B[i][k]
    
    res = [0] * N
    for i in range(N):
        res[i] = B[i][N]
    return res

a, b, c, d, e, f = MAP()

A = [
    [a, b],
    [d, e],
]
B = [c, f]
res = gauss_jordan(A, B)
print(*res)
