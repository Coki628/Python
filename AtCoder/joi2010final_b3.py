# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/joi2011ho/submissions/3889859
　　　https://qnighy.hatenablog.com/entry/20110214/1297649653
・蟻本演習2-3-14
・ソートしてからナップサックDP
・ジャンル別に分けておいて、ソートして前計算。
・各ジャンル単体での個数であれば、追加分を含んだ合計を前もって割り出せる。
・メインのDP部分は1つ目の添字にN全体ではなくジャンルを使う。
・numpy実装。この形はうまくはまった。pythonAC0.2秒。
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

N, K = MAP()
genre = [[] for i in range(10)]
for i in range(N):
    c, g = MAP()
    genre[g-1].append(c)
# 各ジャンル毎に降順ソートしておく
for i in range(10):
    genre[i].sort(reverse=True)

# 各ジャンルi毎にj冊売った時の合計価格を出しておく
V = np.zeros((10, N+1), dtype=np.int64)
for i in range(10):
    for j in range(N):
        if j >= len(genre[i]):
            break
        V[i,j+1] = V[i,j] + genre[i][j] + j * 2

# dp[i][j] := i個目のジャンルまで考えて、合計でj冊売却する場合の価格の最大値
dp = np.zeros((11, K+1), dtype=np.int64)
for i in range(10):
    for k in range(len(genre[i])+1):
        if k > K:
            break
        # ジャンルiについてk冊売った遷移を一括処理する
        dp[i+1,k:] = np.maximum(dp[i+1,k:], dp[i,:K+1-k]+V[i,k])
print(dp[10,K])
