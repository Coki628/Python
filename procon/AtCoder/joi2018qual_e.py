# -*- coding: utf-8 -*-

"""
・蟻本演習3-4-8、自力AC！
・DP、事前計算
・セグ木要るかと思ったけど累積minで大丈夫だった。
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

N, M = MAP()
A = [0] + LIST()
# ある位置を使う場合、その手前の使用位置をどこまで遡って使っていいか事前計算
prev = list(range(N))
for i in range(M):
    l, r = MAP()
    l -= 1; r -= 1
    prev[r] = min(prev[r], l)
# 後ろから累積min
for i in range(N-2, -1, -1):
    prev[i] = min(prev[i], prev[i+1])
prev = [0] + prev

dp = [0] * (N+1)
for i in range(1, N+1):
    # 位置iの電飾を使わないor使うの良い方
    dp[i] = max(dp[i-1], dp[prev[i]] + A[i])
print(dp[N])
