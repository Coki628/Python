# -*- coding: utf-8 -*-

"""
参考：https://ferin-tech.hatenablog.com/entry/2017/11/13/232545
　　　http://kmjp.hatenablog.jp/entry/2014/09/21/0900
・蟻本演習3-4-3
・bitDP、桁DP
・実装疲れた。遷移の時に足す差分の扱いに悩んだ。
　結果的に、作ろうとしている数がAより小さいか大きいかで決まる、だった。
　Aより小さいと、次の桁でaが大きいほど離れるし、dが大きいほど近づく。Aより大きいなら逆。
・各値は遷移の時に*10すると、次桁でも帳尻が合う。
・最上位桁の0を使用済にしない、もコーナーケースにかかった。これは結構みんなやってた。
・計算量はループ一番深いとこでも15*2^10*10=約15万、pythonAC0.5秒。
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

def bit_count(i):

    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0f0f0f0f
    i = i + (i >> 8)
    i = i + (i >> 16)
    return i & 0x3f

A, K = MAP()
A = str(A)
N = len(A)

# dp[i][S][j] := i桁目まで見て、既に使った数の集合をSとして、
# 作ろうとしている数がAと等しい(j=0)かAより小さい(j=1)か大きい(j=2)状態でのAとの最小距離
dp = list3d(N+1, 1<<10, 3, INF)
dp[0][0][0] = 0
for i in range(N):
    a = int(A[i])
    for S in range(1<<10):
        # Kより多く使った状態から遷移させる必要はない
        if bit_count(S) > K:
            continue
        # 各桁各集合から、0~9を全通り試す
        for d in range(10):
            # Aと等しい
            j = 0
            # 集合Sにdが既に含まれているかどうかで遷移先の集合が変わる + 最上位桁の0は使用済にしない
            if (S & (1<<d)) or (S == 0 and d == 0):
                # 今回の桁では同じ数を使う
                if d == a:
                    dp[i+1][S][0] = min(dp[i+1][S][0], dp[i][S][j]*10)
                # Aより小さい数を使う
                elif d < a:
                    dp[i+1][S][1] = min(dp[i+1][S][1], dp[i][S][j]*10 + (a - d))
                # Aより大きい数を使う
                else:
                    dp[i+1][S][2] = min(dp[i+1][S][2], dp[i][S][j]*10 + (d - a))
            else:
                if d == a:
                    dp[i+1][S|1<<d][0] = min(dp[i+1][S|1<<d][0], dp[i][S][j]*10)
                elif d < a:
                    dp[i+1][S|1<<d][1] = min(dp[i+1][S|1<<d][1], dp[i][S][j]*10 + (a - d))
                else:
                    dp[i+1][S|1<<d][2] = min(dp[i+1][S|1<<d][2], dp[i][S][j]*10 + (d - a))
            # 既にAより小さい
            j = 1
            if (S & (1<<d)) or (S == 0 and d == 0):
                dp[i+1][S][j] = min(dp[i+1][S][j], dp[i][S][j]*10 + (a - d))
            else:
                dp[i+1][S|1<<d][j] = min(dp[i+1][S|1<<d][j], dp[i][S][j]*10 + (a - d))
            # 既にAより大きい
            j = 2
            if (S & (1<<d)) or (S == 0 and d == 0):
                dp[i+1][S][j] = min(dp[i+1][S][j], dp[i][S][j]*10 + (d - a))
            else:
                dp[i+1][S|1<<d][j] = min(dp[i+1][S|1<<d][j], dp[i][S][j]*10 + (d - a))

ans = INF
for S in range(1<<10):
    if bit_count(S) <= K:
        ans = min(ans, min(dp[N][S]))
print(ans)
