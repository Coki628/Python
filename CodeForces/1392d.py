"""
・ここまで自力AC
・円環、状態持ってDP
・題意を噛み砕くと、同じ方向に連続3人は無理、が条件となる。
　これは前の2人の方向が分かれば、自分が向ける方向が定まる。
・最初の2人の方向を4通りを決め打って、その先をDPする。
　これで最後に取れる場所も、最初との比較で分かる。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    S = input()

    ans = INF
    # 0: LL, 1: LR, 2: RL, 3: RR
    dp = list2d(4, N+1, INF)
    # from LL
    dp[0][2] = S[:2].count('R')
    for i in range(2, N):
        # LR <- LL
        dp[1][i+1] = min(dp[1][i+1], dp[0][i] + (S[i] != 'R'))
        # RL <- RR
        dp[2][i+1] = min(dp[2][i+1], dp[3][i] + (S[i] != 'L'))
        # RL <- LR
        dp[2][i+1] = min(dp[2][i+1], dp[1][i] + (S[i] != 'L'))
        # RR <- LR
        dp[3][i+1] = min(dp[3][i+1], dp[1][i] + (S[i] != 'R'))
        # LR <- RL
        dp[1][i+1] = min(dp[1][i+1], dp[2][i] + (S[i] != 'R'))
        # LL <- RL
        dp[0][i+1] = min(dp[0][i+1], dp[2][i] + (S[i] != 'L'))
    ans = min(ans, dp[1][N], dp[3][N])

    dp = list2d(4, N+1, INF)
    # from LR
    dp[1][2] = (S[0] != 'L') + (S[1] != 'R')
    for i in range(2, N):
        dp[1][i+1] = min(dp[1][i+1], dp[0][i] + (S[i] != 'R'))
        dp[2][i+1] = min(dp[2][i+1], dp[3][i] + (S[i] != 'L'))
        dp[2][i+1] = min(dp[2][i+1], dp[1][i] + (S[i] != 'L'))
        dp[3][i+1] = min(dp[3][i+1], dp[1][i] + (S[i] != 'R'))
        dp[1][i+1] = min(dp[1][i+1], dp[2][i] + (S[i] != 'R'))
        dp[0][i+1] = min(dp[0][i+1], dp[2][i] + (S[i] != 'L'))
    ans = min(ans, dp[1][N], dp[2][N], dp[3][N])

    dp = list2d(4, N+1, INF)
    # from RL
    dp[2][2] = (S[0] != 'R') + (S[1] != 'L')
    for i in range(2, N):
        dp[1][i+1] = min(dp[1][i+1], dp[0][i] + (S[i] != 'R'))
        dp[2][i+1] = min(dp[2][i+1], dp[3][i] + (S[i] != 'L'))
        dp[2][i+1] = min(dp[2][i+1], dp[1][i] + (S[i] != 'L'))
        dp[3][i+1] = min(dp[3][i+1], dp[1][i] + (S[i] != 'R'))
        dp[1][i+1] = min(dp[1][i+1], dp[2][i] + (S[i] != 'R'))
        dp[0][i+1] = min(dp[0][i+1], dp[2][i] + (S[i] != 'L'))
    ans = min(ans, dp[0][N], dp[1][N], dp[2][N])

    dp = list2d(4, N+1, INF)
    # from RR
    dp[3][2] = S[:2].count('L')
    for i in range(2, N):
        dp[1][i+1] = min(dp[1][i+1], dp[0][i] + (S[i] != 'R'))
        dp[2][i+1] = min(dp[2][i+1], dp[3][i] + (S[i] != 'L'))
        dp[2][i+1] = min(dp[2][i+1], dp[1][i] + (S[i] != 'L'))
        dp[3][i+1] = min(dp[3][i+1], dp[1][i] + (S[i] != 'R'))
        dp[1][i+1] = min(dp[1][i+1], dp[2][i] + (S[i] != 'R'))
        dp[0][i+1] = min(dp[0][i+1], dp[2][i] + (S[i] != 'L'))
    ans = min(ans, dp[0][N], dp[2][N])

    print(ans)
