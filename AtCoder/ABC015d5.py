# -*- coding: utf-8 -*-

"""
・蟻本演習、自力AC
・これも1方向ナップザック系DPの添字増えた系。
・せっかくなのでnumpy実装
・やっぱりこの一括遷移の型にはまると速い。計算量2500万でも0.3秒AC。
"""

import sys
import numpy as np

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
INF = float('inf')
MOD = 10 ** 9 + 7

W = INT()
N, K = MAP()
AB = []
for i in range(N):
    a, b = MAP()
    AB.append((a, b))

dp = np.zeros((N+1, K+1, W+1), dtype=np.int64)
for i in range(N):
    a, b = AB[i]
    for k in range(K+1):
        # 枝刈り
        if k > i:
            break
        # numpyで一括遷移
        dp[i+1,k] = np.maximum(dp[i+1,k], dp[i,k])
        if k != K and a <= W:
            dp[i+1,k+1,a:] = np.maximum(dp[i+1,k+1,a:], dp[i,k,:W+1-a]+b)
# numpyで2,3次元目から一括取得
print(dp[N].max())
