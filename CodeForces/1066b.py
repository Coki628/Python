# -*- coding: utf-8 -*-

"""
・自力AC
・2重ループOKの制約だったので、2次元DPを考えてみたらうまくいった。
・想定解はもっと賢くやってたぽかったけど。。
"""

import sys
from itertools import accumulate

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, r = MAP()
A = [0] + LIST()

B = []
for i, a in enumerate(A):
    if a:
        B.append(i)
M = len(B)

# dp[i][j] := i個目のヒーターまで見て、ヒーターをj個使っている時に、暖が届く最大右端位置
dp = list2d(M+1, M+1, -INF)
dp[0][0] = 0
for i, b in enumerate(B):
    for j in range(M+1):
        # このヒーターを使わない
        dp[i+1][j] = max(dp[i+1][j], dp[i][j])
        # このヒーターが届く左端まで暖が来ていれば、これを使って右端を伸ばせる
        if dp[i][j] >= b - r:
            dp[i+1][j+1] = max(dp[i+1][j+1], b + r - 1)
ans = INF
for j in range(M+1):
    if dp[M][j] >= N:
        ans = j
        break
if ans == INF:
    print(-1)
else:
    print(ans)
