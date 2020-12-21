# -*- coding: utf-8 -*-

# 尺取法

import numpy as np

N,K = map(int, input().split())
sN = [int(input()) for i in range(N)]
sN = np.array(sN)

ans,l,r = 0,0,0
# 左を動かすループ
while l < N:
    # 0があったら合計は絶対0(K以下)にできるので例外処理
    if sN[l] == 0:
        print(N)
        exit()
    # 右を動かすループ
    while r <= N and np.prod(sN[l:r]) <= K:
        r += 1
    ans = max(ans, r - l - 1)
    if l == r:
        r += 1
    l += 1
print(ans)