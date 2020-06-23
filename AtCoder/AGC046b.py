"""
参考：https://twitter.com/shiro537/status/1274349252165029888
　　　https://atcoder.jp/contests/agc046/submissions/14500684
・自力ならず。。
・数え上げDP
・重複分を引きながら更新していく。
・これはうまく作れなかったなー。。
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
MOD = 998244353

a, b, c, d = MAP()
dp = list2d(c+1, d+1, 0)

dp[a][b] = 1
for i in range(a, c+1):
    for j in range(b, d+1):
        # 行を増やす遷移
        dp[i][j] += dp[i-1][j] * j
        # 列を増やす遷移
        dp[i][j] += dp[i][j-1] * i
        # 上から来た分と左から来た分の重複分を除く
        dp[i][j] -= dp[i-1][j-1] * (i-1) * (j-1)
        dp[i][j] %= MOD
ans = dp[c][d]
print(ans)
