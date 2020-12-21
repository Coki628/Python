# -*- coding: utf-8 -*-

""" 解説動画を参考にDPにしてみた版 """

N = int(input())

T = []
A = []
for i in range(N):
    t,a = map(int, input().split())
    T.append(t)
    A.append(a)

# 比率を何倍したら実際の値なのかでDPを埋めてく
dp = [0] * (N)
# 始めの基準を1倍とする
dp [0] = 1

for i in range(1,N):
    # 今回 * 前回 の計算が終わったあとで整数未満を切り上げる
    dp[i] = int(-(-(max(T[i-1] / T[i], A[i-1] / A[i]) * dp[i-1]) // 1))

print((T[-1]+A[-1]) * dp[-1])