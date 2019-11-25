# -*- coding: utf-8 -*-

"""
・蟻本演習3-4-4
・行列累乗
・結構覚えてたけど復習と理解を深めるためにもう1回。
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
                # +の代わりに^、*の代わりに&
                res[i][j] ^= A[i][k] & B[k][j]
    return res

K, M = MAP()
A = LIST()

# &の単位元(全ビット1埋め)
intv = (1<<32) - 1

# 必要な行列を構築
B = [None] * K
B[0] = LIST()
for i in range(1, K):
    B[i] = [0] * K
for i in range(1, K):
    B[i][i-1] = intv

# 例外処理
if M <= K:
    print(A[M-1])
    exit()

# 行列累乗でN項先へ
N = M - K
ans = list2d(K, K, 0)
for i in range(K):
    ans[i][i] = intv
# 繰り返し二乗法
while N > 0:
    if N & 1:
        ans = mat_dot(ans, B)
    B = mat_dot(B, B)
    N >>= 1

# 最後にAと掛ける
ans = mat_dot(ans, A[::-1])
print(*ans[0])
