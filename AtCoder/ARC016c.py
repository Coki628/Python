# -*- coding: utf-8 -*-

"""
参考：https://kimiyuki.net/blog/2016/04/28/dice-and-expected-value/
　　　https://math314.hateblo.jp/entry/2013/12/12/232045
・蟻本演習3-4-2、自力WA
・bitDP、期待値
"""

import sys
from itertools import product

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
INF = float('inf')
MOD = 10 ** 9 + 7

def calc(A):
    # dp[S] := まだ出ていない面の集合Sの状態から始めて、残る全ての面が出るまでに振る回数の期待値
    dp = [0] * (1<<N)
    for bit in range(1, 1<<N):
        tmp = 0
        for i in range(N):
            if bit & (1<<i):
                tmp += A[i]
        if tmp == 0:
            dp[bit] = INF
        else:
            # まだ出ていない面が出るまでに振る回数
            dp[bit] += 1 / tmp
            # 部分集合から必要な値を持ってきて足す
            for i in range(N):
                if bit & (1<<i):
                    if dp[bit^(1<<i)] == INF:
                        dp[bit] = INF
                    else:
                        # それ以降に振る必要のある回数
                        dp[bit] += A[i] / tmp * dp[bit^(1<<i)]
    return dp

N, M = MAP()

P = list2d(M, N, 0)
C = [0] * M
for i in range(M):
    m, cost = MAP()
    C[i] = cost
    for j in range(m):
        a, p = MAP()
        a -= 1
        P[i][a] = p / 100

E = []
for i in range(M):
    E.append(calc(P[i]))

ans = INF
for prod in product(range(M), repeat=N):
    B = [0] * M
    for i, a in enumerate(prod):
        B[a] |= 1<<i
    sm = 0
    for i, b in enumerate(B):
        if E[i][b] == INF:
            sm = INF
            break
        sm += E[i][b] * C[i]
    ans = min(ans, sm)
print(ans)
