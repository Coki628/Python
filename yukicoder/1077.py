"""
・自力AC！
・貰うDPを高速化
・制約から2乗のDPを予想。考えを進めると、まともにやったら遷移がNでつらい。
　愚直を書いてみると、貰うDPの遷移での値の増加量は遷移元に依らない、と分かるので、
　遷移元となる場所から予め最適な値を集めておければOK、今回は累積minする。
・1000万配列のメモリは若干心配だったが、節約やらなくても158MBくらいで問題なし。(制約512MB)
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
M = max(A)

dp = list2d(N+1, M+1, INF)
dp[0][0] = 0
for i, a in enumerate(A):
    # for j in range(M+1):
    #     for k in range(j+1):
    #         dp[i+1][j] = min(dp[i+1][j], dp[i][k] + abs(a-j))
    # 累積minで前計算
    for j in range(M):
        dp[i][j+1] = min(dp[i][j+1], dp[i][j])
    for j in range(M+1):
        # 貰うDP、dp[i][j]は0~jでの最小値になっているので遷移は1個でOK
        dp[i+1][j] = dp[i][j] + abs(a-j)
ans = min(dp[N])
print(ans)
