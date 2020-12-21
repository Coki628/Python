"""
参考：https://drken1215.hatenablog.com/entry/2019/03/20/202800
　　　https://drken1215.hatenablog.com/entry/2019/03/20/173300
・連立方程式の解、掃き出し法
・ガウス・ジョルダンをXOR(というかF2体っていうのか)に適用させた。
・各ビットを行、配列の各値を列に取って、=で結ぶ右辺に当たる値(今回はK)を最終列に置く。
・N-rankは自由に設定できる値があることになるので、XORやMOD 2の世界では2冪して
　2^(N-rank)個の解が存在することになる。
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

N, K = MAP()
A = LIST(N)

# 行にビット、列に値を割り当てる
M = max(A).bit_length()
mat = list2d(M, N+1, 0)
for k in range(M):
    for i in range(N):
        if A[i]>>k & 1:
            mat[k][i] = 1
    # 最後の列にKの値を入れる
    if K>>k & 1:
        mat[k][N] = 1

rank, res = gauss_jordan(mat, extended=True)
if rank == -1:
    print(0)
else:
    ans = pow(2, N-rank, MOD)
    print(ans)
