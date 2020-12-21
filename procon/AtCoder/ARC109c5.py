"""
・前からDPでもやってみた。これだとボトムアップに進める感じ。
　自分ではこっちに近いことをイメージしようとしてたけど、
　辿り着けなかったのは頭が配るDPだったからだと思う。こういうのはきっと配ろうとするとダメで、
　1回戦前の結果2つから貰ってくるDPでイメージするとうまくいった。
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

def check(a, b):
    if a == b:
        return a
    if a == 'R' and b == 'S' \
            or a == 'S' and b == 'P' \
            or a == 'P' and b == 'R':
        return a
    else:
        return b

N, K = MAP()
S = input()

dp = list2d(K+1, N, '')
for i in range(N):
    dp[0][i] = S[i]
for i in range(K):
    for j in range(N):
        dp[i+1][j] = check(dp[i][j], dp[i][(j+pow(2, i))%N])
ans = dp[K][0]
print(ans)
