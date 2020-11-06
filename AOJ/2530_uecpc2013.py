"""
参考：https://drken1215.hatenablog.com/entry/2019/03/20/202800
・蟻本演習4-1-1
・F2体上の連立一次方程式、ガウス・ジョルダン法
・基本的にはこの前やった1308_icpc2010regionalのノウハウがあれば大丈夫。
　自分が影響してひっくり返る所に対して1を設定する感じ。
・違うのは、可能判定だけじゃなくて通り数なこと。
　これはrankの数だけ値が決まっているので、2値なら2^(N-rank)となる。
・N=50^2=2500もあってのN^3なのできつい。C++でbitset高速化やれば通るみたいだけど、
　pythonじゃビットでやっても無理だった。せめてpypy使えたらなぁ。
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
MOD = 10**9 + 9
EPS = 10**-10

# 2値用(ビット高速化版)
def gauss_jordan(A, extended=False):
    """ ガウス・ジョルダン法(連立方程式の解) """

    N = len(A)
    M = N + 1
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
    return i*W+j

def idtogrid(id):
    return divmod(id, W)

def check(a, b):
    # 縦、横、斜めのいずれかにあるか
    return a[0] == b[0] or a[1] == b[1] or abs(a[0]-b[0]) == abs(a[1]-b[1])

H, W = MAP()
grid = [[]] * H
for i in range(H):
    grid[i] = LIST()

N = H * W
mat = [0] * N
for i in range(N):
    h, w = idtogrid(i)
    if grid[h][w]:
        mat[i] |= 1<<N
    for nh in range(H):
        for nw in range(W):
            if check((h, w), (nh, nw)):
                j = gridtoid(nh, nw)
                mat[i] |= 1<<j

rank, res = gauss_jordan(mat, True)
if rank == -1:
    print(0)
else:
    # rankの数だけ値が決まっているので、2値なら通り数は2^(N-rank)
    ans = pow(2, N-rank, MOD)
    print(ans)
