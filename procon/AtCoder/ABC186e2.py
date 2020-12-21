"""
参考：https://tjkendev.github.io/procon-library/python/math/baby-step-giant-step.html
　　　https://atcoder.jp/contests/abc186/submissions/18904196
　　　https://twitter.com/tatyam_prime/status/1340295506140909569
・Baby-Step Giant-Step (BSGS)
・本来は指数部分を求めるのに使うみたいだけど、掛け算してるところを足し算にすると、
　係数部分を求めるのにも使えるみたい。
・でも計算量がlogじゃなくて√になっちゃうし、こっちのが有利になる場面があるのかは分からない。
・ともあれ昔よく分からなくて放置したアルゴを1個回収できたのでOK。
"""

import sys
from math import gcd

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**6)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

# # X^K ≡ Y (mod M) となるような K を求める
# def bsgs(X, Y, M):
#     D = {1: 0}

#     sq = int(M**.5)+1

#     # Baby-step
#     Z = 1
#     for i in range(sq):
#         Z = Z * X % M
#         D[Z] = i+1

#     if Y in D:
#         return D[Y]

#     # Giant-step
#     R = pow(Z, M-2, M) # R = X^(-sq)
#     for i in range(1, sq+1):
#         Y = Y * R % M
#         if Y in D:
#             return D[Y] + i*sq
#     return -1

# K*X ≡ Y (mod M) となるような K を求める？
def bsgs(X, Y, M):
    D = {0: 0}

    sq = int(M**.5)+1

    # Baby-step
    Z = 0
    for i in range(sq):
        Z = (Z + X) % M
        D[Z] = i+1

    if Y in D:
        return D[Y]

    # Giant-step
    R = -X*sq
    for i in range(1, sq+1):
        Y = (Y + R) % M
        if Y in D:
            return D[Y] + i*sq
    return -1

for _ in range(INT()):
    N, S, K = MAP()

    g = gcd(gcd(N, S), K)
    N //= g
    S //= g
    K //= g

    res = bsgs(K, N-S, N)
    print(res)
