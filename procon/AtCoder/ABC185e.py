"""
・自力ならず。これは悔しい。。
・編集距離(レーベンシュタイン距離)のDP
・LCSのDPっぽさはすごく感じていたのに、編集距離のDPを思い出せなかった。。
　うまく話が文字列じゃなくて数列になってるけど、ようは編集距離そのまんまなのね。。。
・アドホック解けないのは多少はしょうがないってなるけど、
　こういう典型そのまま落としちゃうとマジでへこむわー。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
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

N, M = MAP()
A = LIST()
B = LIST()

# dp[i][j] := Aのi番目までとBのj番目までを一致させるのにかかる最小操作数
dp = list2d(N+1, M+1, INF)
dp[0][0] = 0
for i in range(N+1):
    for j in range(M+1):
        # Aのi番目を使わない
        if i < N:
            dp[i+1][j] = min(dp[i+1][j], dp[i][j] + 1)
        # Bのj番目を使わない
        if j < M:
            dp[i][j+1] = min(dp[i][j+1], dp[i][j] + 1)
        # Aのi番目とBのj番目を使う
        if i < N and j < M:
            # 今回増やす値が一致するなら、ノーコストで遷移できる
            dp[i+1][j+1] = min(dp[i+1][j+1], dp[i][j] + (A[i] != B[j]))
ans = dp[N][M]
print(ans)
