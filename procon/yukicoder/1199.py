"""
・さくっと自力AC
・偶奇で状態分けてDPすればOK。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N, M = MAP()
A = [0] * N
for i in range(N):
    A[i] = sum(LIST())

dp = list2d(2, N+1, 0)
for i in range(N):
    dp[0][i+1] = max(dp[0][i+1], dp[0][i])
    dp[1][i+1] = max(dp[1][i+1], dp[1][i])
    dp[1][i+1] = max(dp[1][i+1], dp[0][i] + A[i])
    dp[0][i+1] = max(dp[0][i+1], dp[1][i] - A[i])
ans = max(dp[0][N], dp[1][N])
print(ans)
