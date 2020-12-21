"""
・自力AC
・DP、累積和で前計算する高速化
・累積和と貰うDPで遷移を高速化するやつ。久しぶりにやった。
・累積和にした時添字を1つずらしてるので、遷移考える時ややこしかった。
　今回の場合はif文場合分けがあってもindexedそのままのがやりやすかったかも。
・計算量は300^3=2700万、pypyAC0.68秒。
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

N, D, K = MAP()

dp = list2d(N+1, K+1, 0)
dp[0][0] = 1
for i in range(N):
    acc = [0] * (K+1)
    for j in range(K):
        acc[j+1] = (acc[j] + dp[i][j]) % MOD
    for j in range(K+1):
        l = max(0, j-D)
        r = j
        dp[i+1][j] = (acc[r] - acc[l]) % MOD
ans = dp[N][K]
print(ans)
