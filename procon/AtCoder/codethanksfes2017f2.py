# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/code-thanks-festival-2017-open/editorial.pdf
・XOR系
・数え上げDP
・公式解1：ソートする。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N,K=MAP()
A=[INT() for i in range(N)]
mx=2*max(A)
A.sort()

# dp[i][j] := Aをi番目まで使って、XORがjとなる整数の選び方の個数
dp=list2d(N+1, mx+1, 0)
dp[0][0]=1
limit=0

for i in range(N):
    for j in range(limit+1):
        if dp[i][j]!=0:
            # A[i]を使わない
            dp[i+1][j]=(dp[i+1][j]+dp[i][j])%MOD
            # A[i]を使う
            # iまででjとなった数を、i+1でjとA[i]をXORした値の場所に配る
            dp[i+1][j^A[i]]=(dp[i+1][j^A[i]]+dp[i][j])%MOD
    # 小さい値から論理和を取っていって、徐々にjの範囲を広げる
    limit|=A[i]
print(dp[N][K])
