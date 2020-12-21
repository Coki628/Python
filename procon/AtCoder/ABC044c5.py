# -*- coding: utf-8 -*-

"""
参考：http://arc060.contest.atcoder.jp/data/arc/060/editorial.pdf
・蟻本演習3-1-4
・平均を総和にするテク
・このテクは二分探索だけでなく、DP数え上げでも使えるという例。
・j枚取って..の添字が要らなくなるので、ループを1つ減らせる。
・計算量はN^2*K=50^3=12万5000でpythonも余裕の0.2秒AC。
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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, K = MAP()
# 予め-Kした状態で用意しておく
A = [a-K for a in LIST()]

MAX = N * K * 2
# dp[i][k] := i枚目までのカードで、合計をkにする通り数
dp = list2d(N+1, MAX+1, 0)
# 負の数を考慮して、N*K位置を0とする
dp[0][N*K] = 1
for i in range(N):
    a = A[i]
    for k in range(MAX+1):
        dp[i+1][k] += dp[i][k]
        if 0 <= k+a <= MAX:
            dp[i+1][k+a] += dp[i][k]
# N回終わって合計が0になっている通り数 - 1枚も取らないケース
print(dp[N][N*K]-1)
