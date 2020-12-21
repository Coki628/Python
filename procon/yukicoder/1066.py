"""
・さくっと自力AC！
・2次元DP、数え上げDP
・色がたくさんあるけど、結局興味があるのは、色1の玉を取ったかどうかだけなので、
　色1の玉を取るorそれ以外を取る、の2つの遷移でDPを作れる。
・状態としてはどこまで見たかと、色1を何個取ったかが持てればいいので、N^2がOKなら大丈夫。
　今回の制約がN<=6000の2乗で3600万と厳しめだけど、メモリ節約すれば速度は間に合う。
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
MOD = 998244353
EPS = 10**-10

N, M = MAP()
A = LIST()
B = LIST()

dp = list2d(2, N+1, 0)
dp[0][0] = 1
for i, a in enumerate(A):
    for j in range(N+1):
        # 箱iから色1を取らない
        dp[(i+1)&1][j] += dp[i&1][j] * (a-1)
        dp[(i+1)&1][j] %= MOD
        if j+1 <= N:
            # 箱iから色1を取る
            dp[(i+1)&1][j+1] += dp[i&1][j]
            dp[(i+1)&1][j+1] %= MOD
    dp[i&1] = [0] * (N+1)

for b in B:
    print(dp[N&1][b])
