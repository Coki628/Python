# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/momohara_kyopro/status/1246114542062452736
　　　https://codeforces.com/contest/1330/submission/75395413
・終了後、自力やろうとしたけどダメだった。。
・公式解説よく分かんなくて、ももはらさんのツイートとコードから解読。
・数え上げ、状態をまとめてDP
・最大ビット毎に同じ動きになるので、状態をまとめてDPする。
・2冪毎にまとめられそう、まで考えたのに、うまく作れなかった。。
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

for _ in range(INT()):
    D, MOD = MAP()
    N = D.bit_length()
    # bitln[i] := 最大ビットi毎の、使用できる値の数
    bitln = [0] * (N+1)
    for i in range(1, N+1):
        if D >= 2**i:
            bitln[i] = 2**(i-1)
        else:
            need = D - 2**(i-1) + 1
            bitln[i] = need
            break
    
    # dp[i][j] := 数列をi項目まで考えて、最後の値の最大ビットがiの時の通り数
    dp = list2d(N+1, N+1, 0)
    dp[0][0] = 1
    for i in range(N):
        for j in range(N+1):
            # 最大ビットが現在より大きい数にのみ遷移
            for k in range(j+1, N+1):
                # 前計算した値を使って最大ビット毎にまとめて遷移させる
                dp[i+1][k] += dp[i][j] * bitln[k]
                dp[i+1][k] %= MOD

    ans = 0
    for i in range(1, N+1):
        ans += sum(dp[i])
        ans %= MOD
    print(ans)
