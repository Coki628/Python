# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/otemae2019/editorial-D.pdf
　　　https://atcoder.jp/contests/otemae2019/submissions/6606188
・数え上げDP、桁DP系
・各桁について、mod 3の値毎に数えて集めていく。(3の倍数→桁和が3の倍数であるため)
・遷移の条件には、上記「3の倍数→桁和が3の倍数」と「5の倍数→1の位が0か5」をうまく使う。
・これはN=Mの部分点解。
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

N, M = MAP()
S = [input() for i in range(M)]

if N != M:
    raise Exception

# dp[i][j] := 上からi桁まで見て、桁和がmod 3でjの時の通り数
dp = list2d(N+1, 3, 0)
dp[0][0] = 1

for i in range(N):
    # 3の倍数かつ5の倍数
    if S[i] == 'FizzBuzz':
        # 遷移先は3の倍数(j=0)かつ5の倍数(今回の桁が0か5)の場合
        # (mod)0 + 0 => (mod)0への遷移
        if i != 0:
            dp[i+1][0] += dp[i][0]
        # (mod)1 + 5 => (mod)0への遷移
        dp[i+1][0] += dp[i][1]
        dp[i+1][0] %= MOD
    # 5の倍数(かつ3の倍数ではない)
    elif S[i] == 'Buzz':
        # 遷移先は3の倍数ではない(j=1,2)かつ5の倍数(今回の桁が0か5)の場合
        for j in range(1, 3):
            # 今回の桁が0
            if i != 0:
                dp[i+1][j] += dp[i][j]
            # 今回の桁が5
            dp[i+1][j] += dp[i][(j+1)%3]
            dp[i+1][j] %= MOD
    # 3の倍数(かつ5の倍数ではない)
    elif S[i] == 'Fizz':
        # 0,5以外を使って、(mod)0に遷移させる
        for j in range(10):
            if j == 0 or j== 5:
                continue
            dp[i+1][0] += dp[i][(0-j)%3]
            dp[i+1][0] %= MOD

print(sum(dp[N])%MOD)
