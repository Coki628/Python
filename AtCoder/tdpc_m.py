# -*- coding: utf-8 -*-

"""
・蟻本演習3-4-5、自力AC！
・行列累乗、bitDP(巡回セールスマンちょっと応用)
・ちゃんと遷移がどうなるかを考えて、行列を正しく埋めれて良かった。
・TSPも最短距離からうまく数え上げに適用されられて良かった。
・計算量はbitDPパートのが重くて、2^16*16^3=約2億7000万。
・pypyAC3.7秒(この問題制約8秒、優しい)。
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

def mat_dot(A, B):
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
    res=list2d(n1, m2, 0)
    for i in range(n1):
        for j in range(m2):
            for k in range(n2):
                res[i][j] += A[i][k] * B[k][j]
                res[i][j] %= MOD
    return res

def mat_pow(mat, n, k):
    """ n*nの行列matを行列累乗でk乗する """

    res = list2d(n, n, 0)
    for i in range(n):
        res[i][i] = 1
    # 繰り返し二乗法
    while k > 0:
        if k & 1:
            res = mat_dot(res, mat)
        mat = mat_dot(mat, mat)
        k >>= 1
    return res

H, R = MAP()
G = [None] * R
for i in range(R):
    G[i] = LIST()

# 始点xから始めて、ほぼTSPで最短距離じゃなくて通り数を見る
def calc(x):
    # dp[S][i] := 最後にiにいる時の、集合Sになる通り数
    dp = list2d(1<<R, R, 0)
    dp[1<<x][x] = 1
    for S in range(1, (1<<R)-1):
        for i in range(R):
            # 頂点iが集合Sに含まれていない
            if not S>>i & 1: 
                continue
            for j in range(R):
                # 頂点jは既に集合Sに含まれている
                if S>>j & 1: 
                    continue
                # 辺があれば頂点iからjへの移動を遷移させる
                if G[i][j]:
                    dp[S|(1<<j)][j] += dp[S][i]
                    dp[S|(1<<j)][j] %= MOD
    # 終点毎に通り数を集計
    res = [0] * R
    for S in range(1, 1<<R):
        for i in range(R):
            res[i] += dp[S][i]
            res[i] %= MOD
    return res

# 必要な行列を構築
mat = list2d(R, R, 0)
for j in range(R):
    # 始点j毎に、1階分の移動で遷移する終点の通り数を調べる
    res = calc(j)
    for i in range(R):
        # 遷移に合わせて行列を埋めていく
        mat[i][j] = res[i]
        mat[i][j] %= MOD

# 行列累乗でH項先へ
mat = mat_pow(mat, R, H)

# 最後に初期値と掛ける
mat = mat_dot(mat, [1] + [0]*(R-1))
print(*mat[0])
