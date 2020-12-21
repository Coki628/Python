# -*- coding: utf-8 -*-

"""
・さくっと自力AC！
・bitDP
・すごく典型な感じだから知ってればまあ解けるんだろうけど、
　ほぼ悩まずスムーズに実装して通せたから良かったかな。
・計算量2^10*1000=約100万で、pythonAC1.3秒、pypyAC0.3秒。
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

N, M = MAP()
SC = []
for i in range(M):
    s, c = input().split()
    b = 0
    for i, yn in enumerate(s):
        if yn == 'Y':
            b |= 1<<i
    SC.append((b, int(c)))

dp = list2d(M+1, 1<<N, INF)
dp[0][0] = 0
for i, (bit, c) in enumerate(SC):
    for S in range(1<<N):
        dp[i+1][S] = min(dp[i+1][S], dp[i][S])
        dp[i+1][S|bit] = min(dp[i+1][S|bit], dp[i][S] + c)
ans = dp[M][(1<<N)-1]
if ans == INF:
    print(-1)
else:
    print(ans)
