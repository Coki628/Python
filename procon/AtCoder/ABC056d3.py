# -*- coding: utf-8 -*-

"""
・部分集合を調べるために、部分和問題のDPに帰着させる。
・これは3重ループでTLE想定、部分点獲得。
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
INF = float('inf')
MOD = 10 ** 9 + 7

N, K = MAP()
A = LIST()

def check(omit):
    # 部分和DP
    dp = list2d(N+1, K, 0)
    dp[0][0] = 1
    for i in range(1, N+1):
        for j in range(K):
            dp[i][j] |= dp[i-1][j]
            if i-1 != omit and j-A[i-1] >= 0:
                dp[i][j] |= dp[i-1][j-A[i-1]]
    # 今回の値A[omit]を足してKに到達できるような部分和があれば、これは必要
    for j in range(max(0, K-A[omit]), K):
        if dp[N][j]:
            return True
    # なければ不要
    return False

ans = 0
for i in range(N):
    # 不要なものを数える
    if not check(i):
        ans += 1
print(ans)
