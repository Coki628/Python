"""
・自力AC
・連立方程式
・変数2つだけだし、式変形でもよかったんだけど、ちょうど動確もしたかったからガウス・ジョルダン使った。
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

b1, b2, b3 = MAP()

A = list2d(2, 2, 1)
A[0][0] = b2
A[1][0] = b1
B = [0] * 2
B[0] = b3
B[1] = b2

r, d = gauss_jordan(A, B)
ans = round(b3*r + d)
print(ans)
