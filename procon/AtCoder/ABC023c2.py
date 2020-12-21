# -*- coding: utf-8 -*-

"""
numpy使わない版
"""

R,C,K = map(int, input().split())
N = int(input())

# 行R*列Cの部屋(アメの配置を記録する)
RC = [[0] * C for i in range(R)]
for i in range(N):
    r,c = map(int, input().split())
    RC[r-1][c-1] = 1

# 縦合計
c_sum = [0] * C
for i in range(C):
    tmp = 0
    for j in range(R):
        tmp += RC[j][i]
    c_sum[i] = tmp
# 横合計
r_sum = [0] * R
for i in range(R):
    r_sum[i] = sum(RC[i])

ans = 0
for i in range(R):
    for j in range(C):
        total = r_sum[i] + c_sum[j]
        # 重複対策
        if RC[i][j]:
            total -= 1
        if total == K:
            ans += 1
print(ans)
