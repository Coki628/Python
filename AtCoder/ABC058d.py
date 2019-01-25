# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/arc071/editorial.pdf
・各辺を加減する回数には規則性があるので、ループを1つ減らせる。
・N^2かかる全辺の長さの合計はこうすればNでいけるんだなぁ。
・累積和じゃあ途中の辺の長さ足せないからね…。
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)

MOD = 10 ** 9 + 7

N, M = map(int, input().split())
xN = list(map(int, input().split()))
yM = list(map(int, input().split()))

# 辺の長さの合計を出す
x_sm = 0
for i in range(N):
    x_sm = (x_sm + i * xN[i] - (N-i-1) * xN[i]) % MOD
y_sm = 0
for i in range(M):
    y_sm = (y_sm + i * yM[i] - (M-i-1) * yM[i]) % MOD

print(x_sm * y_sm % MOD)
