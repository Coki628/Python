# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・さくっと自力AC！
・桁DP(+ちょっとbitDP)でオーバーキル
・300点だしさすがに想定解じゃないだろうな、と思いつつも他を考えるより早いと思って実装。
・こういう、典型に持ち込む、みたいなのが出来るようになってきてるのはまあ良いことだよね。
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

S = input()
N = len(S)

# dp[i][bit][j] := i桁目まで見て、出現済の3,5,7の集合がbitで、Nより小さいのが確定しているorしていない(j=0,1)状態の場合の数
dp = list3d(N+1, 1<<3, 2, 0)
dp[0][0][0] = 1
for i in range(N):
    a = int(S[i])
    for bit in range(1<<3):
        # 0への遷移
        if bit == 0:
            dp[i+1][bit][1] += dp[i][bit][0]
            dp[i+1][bit][1] += dp[i][bit][1]
        # 3,5,7への遷移
        for k, x in enumerate([3, 5, 7]):
            dp[i+1][bit|1<<k][1] += dp[i][bit][1]
            if a == x:
                dp[i+1][bit|1<<k][0] += dp[i][bit][0]
            elif a > x:
                dp[i+1][bit|1<<k][1] += dp[i][bit][0]
# N桁見終わって、3,5,7出現済(bit=111)の合計
print(dp[N][-1][0] + dp[N][-1][1])
