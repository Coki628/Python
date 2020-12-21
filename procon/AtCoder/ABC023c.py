# -*- coding: utf-8 -*-

import numpy as np

R,C,K = map(int, input().split())
N = int(input())

# 行R*列Cの部屋(アメの配置を記録する)
RC = np.zeros((R,C,))
for i in range(N):
    r,c = map(int, input().split())
    RC[r-1][c-1] = 1
# 縦合計
c_sum = np.sum(RC, axis=0)
# 横合計
r_sum = np.sum(RC, axis=1)

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
