"""
参考：https://betrue12.hateblo.jp/entry/2019/03/18/005750
　　　https://trap.jp/post/435/
・連立方程式の解、掃き出し法
・同じXORのガウス・ジョルダンなんだけど、ちょっと別の方針。
・ここでは比較する右辺は行列に繋げずに、左辺のみを処理した結果に対して確認を取る。
・結果の行列に対して、大きいビットから貪欲にKに合わせて行く。うまくいけば2^(N-rank)個の解、ダメなら解なし。
・詳細はソース内コメントで。
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

# 行に値、列にビットを割り当てる
M = max(A).bit_length()
mat = list2d(N, M, 0)
for i in range(N):
    for k in range(M):
        if A[i]>>k & 1:
            mat[i][k] = 1
    # 大きいビットが前に来るようにリバース
    mat[i].reverse()
rank, res = gauss_jordan(mat, extended=False)
k = M - 1
cur = 0
for i in range(rank):
    # kとの兼ね合いで小さいビットを前に戻す
    res[i].reverse()
    # res[i]の先頭から最初にビットが立っている位置を探す
    while k >= 0 and res[i][k] == 0:
        k -= 1
    # 立っていたビットがKに必要で、現在のcurで立ってなかったら、このres[i]を使う
    if k >= 0 and K>>k & 1 and not cur>>k & 1:
        for j in range(k, -1, -1):
            if res[i][j]:
                cur ^= 1 << j
# curをKと一致させられたらOK、ダメなら解なしで0
if cur == K:
    ans = pow(2, N-rank, MOD)
    print(ans)
else:
    print(0)
