"""
・マラソン、局所探索法(山登り法)の準備。
・これは再計算処理が愚直なのでさすがにTLE。。
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
EPS = 10 ** -10

D = INT()
C = LIST()
S = [[]] * D
for i in range(D):
    S[i]  = LIST()
T = [t-1 for t in LIST(D)]
M = 26

def check(T):
    last = [0] * M
    res = 0
    for i, t in enumerate(T):
        res += S[i][t]
        for j in range(M):
            last[j] += 1
        last[t] = 0
        for j in range(M):
            res -= C[j] * last[j]
    return res

Q = INT()
for i in range(Q):
    d, q = MAP()
    d -= 1; q -= 1

    T[d] = q
    res = check(T)
    print(res)
