# -*- coding: utf-8 -*-

"""
Nから0に降りてくるDP(解説参考)
"""

N = int(input())
NG = [0] * 3
NG[0] = int(input())
NG[1] = int(input())
NG[2] = int(input())

# [地点] = 最短手数 のDP
dp = [float('inf')] * (N+1)
# 開始地点
dp[N] = 0

for i in range(N, 0, -1):
    # NG位置からは移動できないのでスキップ
    if i in NG:
        continue
    # 1～3を引いていく
    for j in range(1, 4):
        if i - j >= 0:
            dp[i-j] = min(dp[i] + 1, dp[i-j])
# 0地点に100手以内で着いたかどうか
if dp[0] <= 100:
    print('YES')
else:
    print('NO')
