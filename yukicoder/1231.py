"""
・さくっと自力AC！ストレートに考察進むとやっぱ気持ちいいね！
・modで状態持つ典型DP
・10の倍数かどうかだけが大事なので、mod 10での値を状態として持つやつ。よく出るやつ。
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

N = INT()
A = LIST()

dp = list2d(N+1, 10, -INF)
dp[0][0] = 0
for i in range(N):
    for j in range(10):
        dp[i+1][j] = max(dp[i+1][j], dp[i][j])
        dp[i+1][(j+A[i])%10] = max(dp[i+1][(j+A[i])%10], dp[i][j] + 1)
ans = dp[N][0]
print(ans)
