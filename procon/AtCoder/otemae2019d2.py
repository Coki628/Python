# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/otemae2019/editorial-D.pdf
　　　https://atcoder.jp/contests/otemae2019/submissions/6606188
・数え上げDP、桁DP系
・各桁について、mod 3の値毎に数えて集めていく。(3の倍数→桁和が3の倍数であるため)
・遷移の条件には、上記「3の倍数→桁和が3の倍数」と「5の倍数→1の位が0か5」をうまく使う。
・叫ばない回を考慮するために添字を1つ追加する。これでAC。
・計算量はO(NM)=1000^2=100万だけど、実際はその内側で最大10*2のループが回るので2000万、
　pythonTLEでpypy1.6秒AC。
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

# dp[i][j][k] := 上からi桁まで見て、桁和がmod 3でjの時で、k回叫んだ状態の通り数
dp = list3d(N+1, 3, M+1, 0)
dp[0][0][0] = 1

for i in range(N):
    for k in range(M+1):
        # 叫ばない分の遷移
        for j in range(10):
            for x in range(1, 3):
                # 遷移先は3の倍数ではない(x=1,2)かつ5の倍数ではない(j=0,5以外)
                if j == 0 or j== 5:
                    continue
                dp[i+1][x][k] += dp[i][(x-j)%3][k]
                dp[i+1][x][k] %= MOD

        # M回叫んだ後は叫ぶ遷移はない
        if k == M:
            continue
        # 3の倍数かつ5の倍数
        if S[k] == 'FizzBuzz':
            # 遷移先は3の倍数(j=0)かつ5の倍数(今回の桁が0か5)の場合
            # (mod)0 + 0 => (mod)0への遷移
            if i != 0:
                dp[i+1][0][k+1] += dp[i][0][k]
            # (mod)1 + 5 => (mod)0への遷移
            dp[i+1][0][k+1] += dp[i][1][k]
            dp[i+1][0][k+1] %= MOD
        # 5の倍数(かつ3の倍数ではない)
        elif S[k] == 'Buzz':
            # 遷移先は3の倍数ではない(j=1,2)かつ5の倍数(今回の桁が0か5)の場合
            for j in range(1, 3):
                # 今回の桁が0
                if i != 0:
                    dp[i+1][j][k+1] += dp[i][j][k]
                # 今回の桁が5
                dp[i+1][j][k+1] += dp[i][(j+1)%3][k]
                dp[i+1][j][k+1] %= MOD
        # 3の倍数(かつ5の倍数ではない)
        elif S[k] == 'Fizz':
            # 0,5以外を使って、(mod)0に遷移させる
            for j in range(10):
                if j == 0 or j== 5:
                    continue
                dp[i+1][0][k+1] += dp[i][(0-j)%3][k]
                dp[i+1][0][k+1] %= MOD

print((dp[N][0][M]+dp[N][1][M]+dp[N][2][M])%MOD)
