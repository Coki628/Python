"""
参考：https://bondo.hateblo.jp/entry/2020/10/21/110119
・自力ならず。。これは解き切りたかったな。
・部分和DP応用
・部分和DPこねくり回す方針はよかったのに詰めきれなかったな。。
・そこまでの和は状態としてはmod Kにまとめていいんだよね、典型考察なのに出なかった。悔しい。
　それがまとまれば、その行で何個使ったかと合わせて両方状態持っても間に合う。
・とはいえ4次元リスト使って70^4/2=約1200万を制約1秒はきついかと持ったけど、
　とりあえずギリギリ0.998秒AC。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N, M, K = MAP()

# dp[i][j][k][l] := i行j列まで見て、その行でk個使ってmod Kがlの状態の最大値
dp = list4d(N+1, M+1, M//2+1, K, -1)
dp[0][0][0][0] = 0
for i in range(N):
    row = LIST()

    for j in range(M):
        for k in range(M//2+1):
            for l in range(K):
                if dp[i][j][k][l] != -1:
                    dp[i][j+1][k][l] = max(dp[i][j+1][k][l], dp[i][j][k][l])
                    if k+1 <= M//2:
                        dp[i][j+1][k+1][(l+row[j])%K] = max(dp[i][j+1][k+1][(l+row[j])%K], dp[i][j][k][l] + row[j])
    
    # i行目での各mod Kでの最大値をi+1行目の初期状態に持っていく
    for k in range(M//2+1):
        for l in range(K):
            dp[i+1][0][0][l] = max(dp[i+1][0][0][l], dp[i][M][k][l])

ans = dp[N][0][0][0]
print(ans)
