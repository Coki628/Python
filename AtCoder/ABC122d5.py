# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・さくっと自力AC！
・数え上げDP。条件に合わせて色々添字増やす系。ぱっと出た方針でそのままキレイに通せた。
・当時は自力じゃ歯が立たなかったぽいので、これは嬉しいね。
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

# dp[i][j][k][l] := i文字目まで見て、3文字前がj、2文字前がk、1文字前がlである通り数
dp = list4d(N+1, 5, 5, 5, 0)
dp[0][0][0][0] = 1

# A: 1, C: 2, G: 3, T: 4
for i in range(N):
    for j in range(5):
        for k in range(5):
            for l in range(5):
                # A,C,G,Tへの遷移
                for m in range(1, 5):
                    # AGC
                    if k == 1 and l == 3 and m == 2:
                        continue
                    # ACG
                    elif k == 1 and l == 2 and m == 3:
                        continue
                    # GAC
                    elif k == 3 and l == 1 and m == 2:
                        continue
                    # A?GC
                    elif j == 1 and l == 3 and m == 2:
                        continue
                    # AG?C
                    elif j == 1 and k == 3 and m == 2:
                        continue
                    dp[i+1][k][l][m] += dp[i][j][k][l]
                    dp[i+1][k][l][m] %= MOD
ans = 0
for j in range(5):
    for k in range(5):
        for l in range(5):
            ans += dp[N][j][k][l]
            ans %= MOD
print(ans)
