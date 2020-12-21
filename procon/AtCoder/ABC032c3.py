# -*- coding: utf-8 -*-

# 尺取法
# (人の参考にして整理した版)

N,K = map(int, input().split())
sN = [int(input()) for i in range(N)]

ans,l,r = 0,0,0
tmp = 1
# 左を動かすループ
while l < N:
    # 0があったら合計は絶対0(K以下)にできるので例外処理
    if sN[l] == 0:
        print(N)
        exit()
    # 右を動かすループ
    while r < N and tmp * sN[r] <= K:
        # *次の数
        tmp *= sN[r]
        r += 1  
    ans = max(ans, r - l)
    # lとrが一緒に動く時は減らさなくていい
    if l == r:
        r += 1
    else:
        # /前の数
        tmp //= sN[l]
    l += 1
print(ans)