"""
参考：https://qiita.com/drken/items/f2ea4b58b0d21621bd51
・写像12相勉強会
・9相
・(第2種)スターリング数
・DPで解く。遷移の意味とかはけんちょんさんの記事に詳しい。
・これだけだと包除のが計算量が軽いが、こちらは応用が効く。
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

# N個の区別する玉を、K個の区別しない箱に入れる(各箱に玉1個以上)
def stirling(N, K):
    dp = list2d(N+1, K+1, 0)
    dp[0][0] = 1
    for i in range(1, N+1):
        for k in range(1, K+1):
            dp[i][k] = dp[i-1][k-1] + k*dp[i-1][k]
            dp[i][k] %= MOD
    return dp[N][K]

N, K = MAP()

ans = stirling(N, K)
print(ans)
