"""
参考：https://drken1215.hatenablog.com/entry/2019/03/20/202800
・連立方程式、掃き出し法、XOR
・ようやく流れを理解した。時間かかったー。これはTLE。
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

for _ in range(INT()):
    N = INT()
    A = LIST()
    S = input()
    M = max(A).bit_length()

    mat = list2d(M, 1, 0)
    for i in range(N-1, -1, -1):
        # 先手番なら今回のA[i]を左辺に取り込む
        if S[i] == '0':
            for k in range(M):
                mat[k].append(A[i]>>k & 1)
        # 後手番なら今回のA[i]を右辺として、解があるか確かめる
        else:
            cur = [x[:] for x in mat]
            for k in range(M):
                cur[k].append(A[i]>>k & 1)
            # これまでの値の集合matでA[i]を作れるかどうか
            rank, res = gauss_jordan(cur, extended=True)
            if rank == -1:
                print(1)
                break
    else:
        print(0)
