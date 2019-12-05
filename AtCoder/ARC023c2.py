# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc023
・練習のため、部分点解法もやってみる。
・数え上げDP、貰うDP、累積和で高速化
"""

import sys
from itertools import accumulate

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

N = INT()
A = [0] + LIST()
mx = max(A)

# dp[i][j] := i番目の数字がjである場合の数
dp = list2d(N+1, mx+1, 0)
dp[0][0] = 1
for i in range(1, N+1):
    a = A[i]
    if a != -1:
        for j in range(mx+1):
            # aより大きい数からは遷移させない
            if j > a:
                break
            # aが確定した数字なら全てそこに遷移
            dp[i][a] += dp[i-1][j]
            dp[i][a] %= MOD
    else:
        # まとめて遷移するための累積和
        acc = list(accumulate(dp[i-1]))
        for j in range(mx+1):
            # 自分以下の数全てから遷移させる
            dp[i][j] += acc[j]
            dp[i][j] %= MOD
print(dp[N][A[-1]])
