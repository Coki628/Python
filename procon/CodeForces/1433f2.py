"""
参考：https://bondo.hateblo.jp/entry/2020/10/21/110119
・自力ならず。。これは解き切りたかったな。
・部分和DP応用
・部分和DPこねくり回す方針はよかったのに詰めきれなかったな。。
・そこまでの和は状態としてはmod Kにまとめていいんだよね、典型考察なのに出なかった。悔しい。
　それがまとまれば、その行で何個使ったかと合わせて両方状態持っても間に合う。
・とはいえ4次元リスト使って70^4/2=約1200万を制約1秒はきついかと持ったけど、
　とりあえずギリギリAC0.998秒。。
・配列使い回しで3次元リストにしてやった。これだとちょっと安心のAC0.76秒。
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
dp = list3d(M+1, M//2+1, K, -1)
nxt = [-1] * K
dp[0][0][0] = 0
for i in range(N):
    row = LIST()

    for j in range(M):
        for k in range(M//2+1):
            for l in range(K):
                if dp[j][k][l] != -1:
                    dp[j+1][k][l] = max(dp[j+1][k][l], dp[j][k][l])
                    if k+1 <= M//2:
                        dp[j+1][k+1][(l+row[j])%K] = max(dp[j+1][k+1][(l+row[j])%K], dp[j][k][l] + row[j])
    
    # i行目での各mod Kでの最大値をi+1行目の初期状態に持っていく
    for k in range(M//2+1):
        for l in range(K):
            nxt[l] = max(nxt[l], dp[M][k][l])
    dp = list3d(M+1, M//2+1, K, -1)
    dp[0][0] = nxt
    nxt = [-1] * K

ans = dp[0][0][0]
print(ans)
