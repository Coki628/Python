# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-1、自力AC！
・桁DP
・Dの倍数の通り数 => mod Dで0になる数の通り数-1 の言い換え考察も大事だと思う。
・計算量はN*D*10=10000*100*10=1000万、pythonTLE、pypyAC0.7秒。
・numpy実装もしてみた。桁DPでも適用できたのは良かったけど、これでもpythonTLE。
・多分、Dのループを速くしたところで今回はN=10000,D=100だから、どっちにしろNのループ*10が重いんだと思う。。
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

D = INT()
S = input()
N = len(S)

dp = np.zeros((2, N+1, D), dtype=np.int64)
dp[0,0,0] = 1
for i in range(N):
    s = int(S[i])
    for k in range(10):
        # 遷移の境界(mod Dで整える)
        a = k % D
        # 3次元目の0〜D-1までをまとめて遷移させる
        if k < s:
            dp[1,i+1,a:] += dp[0,i,:D-a]
            dp[1,i+1,:a] += dp[0,i,D-a:]
            dp[1,i+1,a:] += dp[1,i,:D-a]
            dp[1,i+1,:a] += dp[1,i,D-a:]
            dp[1,i+1] %= MOD
        elif k == s:
            dp[:,i+1,a:] += dp[:,i,:D-a]
            dp[:,i+1,:a] += dp[:,i,D-a:]
            dp[:,i+1] %= MOD
        else:
            dp[1,i+1,a:] += dp[1,i,:D-a]
            dp[1,i+1,:a] += dp[1,i,D-a:]
            dp[1,i+1] %= MOD

print((dp[0,N,0]+dp[1,N,0]-1)%MOD)
