# -*- coding: utf-8 -*-

"""
・自力TLE。。
・2000^2=400万のDPに最大2000文字の文字列は持たせ過ぎだった。
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

D = [
    '1110111', '0010010', '1011101', '1011011', '0111010', '1101011', '1101111', '1010010', '1111111', '1111011',
]
for i in range(10):
    D[i] = int(D[i], 2)

N, K = MAP()
A = [int(input(), 2) for i in range(N)]

dp = list2d(N+1, K+1, None)
dp[0][0] = ''
for i, a in enumerate(A):
    for j in range(K+1):
        if dp[i][j] is None:
            continue
        for x, d in enumerate(D):
            cnt = 0
            ng = 0
            for k in range(7):
                if not a>>k & 1 and d>>k & 1:
                    cnt += 1
                if a>>k & 1 and not d>>k & 1:
                    ng = 1
                    break
            if not ng and j+cnt <= K:
                if dp[i+1][j+cnt] is None:
                    dp[i+1][j+cnt] = dp[i][j] + str(x)
                else:
                    dp[i+1][j+cnt] = max(dp[i+1][j+cnt], dp[i][j] + str(x))
ans = dp[N][K]
if ans is None:
    print(-1)
else:
    print(ans)
