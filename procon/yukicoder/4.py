"""
・さくっと自力AC！
・部分和DP
・全て使ってちょうど半分にしたいので、部分和DPで総和の半分になるものが存在するかチェックすればOK。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
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
MX = 100

sm = sum(A)
if sm % 2 == 1:
    print('impossible')
    exit()

dp = list2d(N+1, N*MX+1, 0)
dp[0][0] = 1
for i in range(N):
    for j in range(N*MX+1):
        if dp[i][j]:
            dp[i+1][j] |= dp[i][j]
            dp[i+1][j+A[i]] |= dp[i][j]

if dp[N][sm//2]:
    print('possible')
else:
    print('impossible')
