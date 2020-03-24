# -*- coding: utf-8 -*-

"""
・自力AC！
・添字にマイナスも管理させたいDP
・端子数が2000以上まで増えたら、その先で足りなくなる心配はないはずなので、
　そこを添字の最大とする。(でないとキー増えすぎて多分爆発する)
・計算量400万、普通にやったらpypyMLEしたので、1つ目の添字を偶奇だけにする。pypyAC0.8秒。
"""

import sys
from collections import defaultdict

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
AB = []
for _ in range(N):
    a, b = MAP()
    AB.append((a, b))

# dp[i][j] := i個目のストラップまで見て、残りの端子がj個の時の最大嬉しさ
dp = [defaultdict(lambda: -INF) for i in range(2)]
dp[0][1] = 0
for i, (a, b) in enumerate(AB):
    for k, v in dp[i&1].items():
        # これを付けない
        dp[(i+1)&1][k] = max(dp[(i+1)&1][k], v)
        # これを付ける
        dp[(i+1)&1][min(k+a-1, 2000)] = max(dp[(i+1)&1][min(k+a-1, 2000)], v + b)
ans = -INF
for k, v in dp[N&1].items():
    # 端子の残数がマイナスでないなら実施可能
    if k >= 0:
        ans = max(ans, v)
print(ans)
