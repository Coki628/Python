# -*- coding: utf-8 -*-

N, W = map(int, input().split())

vN = [0] * N
wN = [0] * N
for i in range(N):
    vN[i], wN[i] = map(int, input().split())

# 応用ナップザック(実は1次元DPでいける)
if max(wN) > 1000:
    # 荷物を全部入れた場合のvalueを内ループに使う
    max_v = sum(vN)
    # 荷物のvalueを添え字にして重さを配列内に格納するDP
    dp = [float('inf')] * (max_v+1)
    dp[0] = 0
    for i in range(N):
        for j in range(max_v, -1, -1):
            if j >= vN[i]:
                dp[j] = min(dp[j-vN[i]] + wN[i], dp[j])
    ans = 0
    # 重さWの範囲内で価値が最大のもの
    for i in range(max_v+1):
        if dp[i] <= W:
            ans = max(ans, i)
    print(ans)

# 普通のナップザック(実はこれも1次元DPでいける)
else:
    # 荷物の重さを添え字にしてvalueを配列内に格納するDP
    dp = [0] * (W+1)
    for i in range(N):
        for j in range(W, -1, -1):
            if j >= wN[i]:
                dp[j] = max(dp[j-wN[i]] + vN[i], dp[j])
    print(dp[W])