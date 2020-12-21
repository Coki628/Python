"""
参考：https://qiita.com/drken/items/f2ea4b58b0d21621bd51
・写像12相勉強会
・10相
・分割数
・蟻本で1回通ってる概念だけど、これ入ってこないよなー。。
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

# 分割数：自然数NをK個の0以上の整数の和で表す場合の数
def partition(N, K):
    # dp[i][j] := jのi分割の総数
    dp = list2d(K+1, N+1, 0)
    dp[0][0] = 1
    for i in range(1, K+1):
        for j in range(N+1):
            dp[i][j] += dp[i-1][j]
            if j - i >= 0:
                dp[i][j] += dp[i][j-i]
            dp[i][j] %= MOD
    return dp[K][N]

N, K = MAP()

ans = partition(N, K)
print(ans)
