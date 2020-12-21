# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・普通な感じのナップザック系DP。最後に答えに使う値だけ、問題よく見る。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

# dp[i][j] := i番目の数まで見て、合計がjとなっている通り数
dp = list2d(N, 21, 0)
dp[1][A[0]] = 1
for i, a in enumerate(A[1:-1], 1):
    for j in range(21):
        # 2番目〜最後から2番目までの値を、有効な範囲内で遷移させる
        if j+a <= 20:
            dp[i+1][j+a] += dp[i][j]
        if j-a >= 0:
            dp[i+1][j-a] += dp[i][j]
# 最後は=を付けるので、最後の値と一致している通り数が答え
ans = dp[N-1][A[-1]]
print(ans)
