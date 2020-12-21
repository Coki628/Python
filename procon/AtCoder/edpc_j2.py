# -*- coding: utf-8 -*-

"""
参考：https://www.hamayanhamayan.com/entry/2016/11/14/223727
　　　https://qiita.com/drken/items/03c7db44ccd27820ea0d
　　　https://kyopro-friends.hatenablog.com/entry/2019/01/12/231000
・期待値DP
・条件付き期待値：(期待値) = sum((条件付き期待値)*(その条件となる確率))
・式変形で遷移を作ってる人が多そうだけど、こっちの考え方で行くと、
　そもそも自己ループへの遷移を意識しなくて良さそう。
・再帰じゃない版。参照順の関係でiとkの添字は逆にした。
・これもTLEするので、枝刈り入れてやっとAC。pypy0.8秒。
"""

import sys
from collections import Counter

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

N = INT()
A = LIST()
C = Counter(A)

# dp[i][j][k] := 3個残りがi個、2個残りがj個、1個残りがk個である状態にするまでの操作回数の期待値
dp = list3d(N+1, N+1, N+1, INF)
# 全て残0なら残り回数は0
dp[0][0][0] = 0.0
for i in range(N+1):
    for j in range(N+1):
        # 枝刈り
        if i+j > N:
            break
        for k in range(N+1):
            # 枝刈り
            if i+j+k > N:
                break
            if i == j == k == 0:
                continue
            # 今現在残っている皿を取る確率
            tmp = (i + j + k) / N
            # そのために必要な回数
            res = 1 / tmp
            # それ以降に必要な回数
            if i and j < N: 
                res += (i/N) / tmp * dp[i-1][j+1][k]
            if j and k < N: 
                res += (j/N) / tmp * dp[i][j-1][k+1]
            if k: 
                res += (k/N) / tmp * dp[i][j][k-1]
            dp[i][j][k] = res
print(dp[C[3]][C[2]][C[1]])
