"""
・さくっと自力AC
・部分和DP
・部分和DPやったばっかだからすぐ見えたけど、しばらくやってないと忘れるんだよな。。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

N, M, K = MAP()
A = [[]] * N
for i in range(N):
    A[i] = LIST()

dp = list2d(N+1, K+1, 0)
dp[0][0] = 1
for i in range(N):
    for j in range(K+1):
        if dp[i][j]:
            for a in A[i]:
                if j+a <= K:
                    dp[i+1][j+a] |= dp[i][j]
ans = -1
for i in range(K+1):
    if dp[N][i]:
        ans = K - i
print(ans)
