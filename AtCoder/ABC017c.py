# -*- coding: utf-8 -*-

"""
いもす法
"""

N,M = map(int, input().split())
L,R,S = [0]*N,[0]*N,[0]*N
for i in range(N):
    L[i],R[i],S[i] = map(int, input().split())
total = sum(S)

# いもすで集計準備
imos = [0] * M
for i in range(N):
    imos[L[i]-1] += S[i]
    # 最後の減少はやらない
    if R[i] != M:
        imos[R[i]] -= S[i]

# その宝石を取らないことによる失点を集計する
memo = [0] * M
memo[0] = imos[0]
for i in range(1, M):
    memo[i] = imos[i] + memo[i-1]

# 全合計-失点の一番少ないケース、が最大値
print(total-min(memo))
