"""
参考：http://satelliteyezu.hatenablog.com/entry/2016/04/27/232439
　　　http://drken1215.hatenablog.com/entry/2019/03/20/160500
　　　http://drken1215.hatenablog.com/entry/2019/03/20/202800
　　　http://www5d.biglobe.ne.jp/~tomoya03/shtml/algorithm/GaussJ.htm
・蟻本演習3-2-4
・反転(ライツアウト)、連立一次方程式、ガウス・ジョルダン法
・蟻本中級埋めで撤退したやつ。年月を越えて、やっと攻略。
・グリッドはマスにID振る感じになるので、変換関数作っとくと楽。
・ガウスジョルダン、今回ビットでやる速い方使ったけど、こっちだと式の向きが直感と逆になるので注意。
　(例えば右辺は最大ビットなので、一番左っぽい感じになる。)
・計算量が(NM)^3=2億4400万くらいあるけど、制約8秒(AOJはpythonだと*5だっけ？)でpythonAC10.2秒。
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

# 2値用(ビット高速化版)
def gauss_jordan(N, M, A, extended=False):
    """ ガウス・ジョルダン法(連立方程式の解) """

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

def gridtoid(i, j):
    return i*M+j

def idtogrid(id):
    return divmod(id, M)

def get_dist(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

while 1:
    M, N, D = MAP()
    if M == N == D == 0:
        break

    grid = [[]] * N
    for i in range(N):
        grid[i] = LIST()

    mat = [0] * (N*M)
    for i in range(N*M):
        h, w = idtogrid(i)
        # 右辺の0 or 1、ここはグリッドの初期値による
        if grid[h][w]:
            mat[i] |= 1<<(N*M)
        # こっから左辺、各マス全探索で距離がDならビットを立てる
        mat[i] |= 1<<i
        for nh in range(N):
            for nw in range(M):
                if get_dist((h, w), (nh, nw)) == D:
                    j = gridtoid(nh, nw)
                    mat[i] |= 1<<j

    rank, res = gauss_jordan(N*M, N*M+1, mat, True)
    if rank == -1:
        print(0)
    else:
        print(1)
