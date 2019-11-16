# -*- coding: utf-8 -*-

"""
・500点自力AC！
・ソートしてナップザック
・蟻本進めてる成果が出た感じ。
・pypyAC0.7秒。
"""

import sys
from operator import itemgetter

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

N, T = MAP()
AB = []
for i in range(N):
    a, b = MAP()
    AB.append((a, b))

# 各料理を時間の昇順でソート
AB.sort(key=itemgetter(1), reverse=True)
AB.sort(key=itemgetter(0))

# 後は普通のナップザック
TMAX = 6007
dp = list2d(N+1, TMAX, 0)
for i in range(N):
    a, b = AB[i]
    for j in range(TMAX):
        dp[i+1][j] = max(dp[i+1][j], dp[i][j])
        if j < T:
            dp[i+1][j+a] = max(dp[i+1][j+a], dp[i][j] + b)
print(max(dp[N]))
