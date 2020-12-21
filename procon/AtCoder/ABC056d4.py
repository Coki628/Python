# -*- coding: utf-8 -*-

"""
・とりあえずメモリ(空間計算量)削減版
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
    dp = [0] * K
    dp[0] = 1
    for i in range(1, N+1):
        for j in range(K-1, -1, -1):
            if i-1 != omit and j-A[i-1] >= 0:
                dp[j] |= dp[j-A[i-1]]
    # 今回の値A[omit]を足してKに到達できるような部分和があれば、これは必要
    for j in range(max(0, K-A[omit]), K):
        if dp[j]:
            return True
    # なければ不要
    return False

ans = 0
for i in range(N):
    # 不要なものを数える
    if not check(i):
        ans += 1
print(ans)
