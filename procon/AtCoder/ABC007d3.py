# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc007
　　　https://torus711.hatenablog.com/entry/20140419/1397919346
　　　https://atcoder.jp/contests/abc007/submissions/6609709
・桁DP
・2種類のフラグに4つの状態がある。うまく遷移させて数え上げる。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

A, B = MAP()

def f(A):
    S = str(A)
    N = len(S)
    # dp[i][j][k] := 大きい方からi桁目まで考えて、jをsmallerフラグ、kを既に4,9が出現したどうか
    # とした時の、その条件を満たす状態の通り数
    dp = list3d(N+1, 2, 2, 0)
    dp[0][0][0] = 1
    # 各桁のループ
    for i, s in enumerate(S):
        num = int(s)
        for j in range(2):
            for k in range(2):
                # 各桁0~9まで遷移する可能性がある
                for l in range(10):
                    # smallerフラグが立っている方に遷移
                    if l < num:
                        # 4,9かどうかによって出現フラグが立っている方に遷移
                        if l == 4 or l == 9:
                            dp[i+1][1][1] += dp[i][j][k]
                        else:
                            dp[i+1][1][k] += dp[i][j][k]
                    # 次桁移行に持ち越し
                    elif l == num:
                        if l == 4 or l == 9:
                            dp[i+1][j][1] += dp[i][j][k]
                        else:
                            dp[i+1][j][k] += dp[i][j][k]
                    # l > num はsmallerフラグが立っているもののみ遷移
                    elif j:
                        if l == 4 or l == 9:
                            dp[i+1][j][1] += dp[i][j][k]
                        else:
                            dp[i+1][j][k] += dp[i][j][k]
    # N桁終わって、出現フラグが立っているものを合わせる
    return dp[N][0][1] + dp[N][1][1]

print(f(B) - f(A-1))
