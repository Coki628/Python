# -*- coding: utf-8 -*-

N, W = map(int, input().split())

vN = [0] * N
wN = [0] * N
for i in range(N):
    vN[i], wN[i] = map(int, input().split())

# 応用ナップザック
if max(wN) > 1000:
    # 荷物を全部入れた場合のvalueを内ループに使う
    max_v = sum(vN)
    # 荷物のvalueを2つめの添え字にして重さを配列内に格納するDP
    dp = [[float('inf')] * (max_v+1) for i in range(N+1)]
    dp[0][0] = 0

    for i in range(N):
        for j in range(max_v+1):
            if j >= vN[i]:
                dp[i+1][j] = min(dp[i][j-vN[i]] + wN[i], dp[i][j])
            else:
                dp[i+1][j] = dp[i][j]
    ans = 0
    # 重さWの範囲内で価値が最大のもの
    for i in range(max_v+1):
        if dp[N][i] <= W:
            ans = max(ans, i)
    print(ans)

# いわゆる普通のナップザック
else:
    # 荷物の重さを2つめの添え字にしてvalueを配列内に格納するDP
    dp = [[0] * (W+1) for i in range(N+1)]

    for i in range(N):
        for j in range(W+1):
            if j >= wN[i]:
                dp[i+1][j] = max(dp[i][j-wN[i]] + vN[i], dp[i][j])
            else:
                dp[i+1][j] = dp[i][j]
    print(dp[N][W])