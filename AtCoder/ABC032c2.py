# -*- coding: utf-8 -*-

# しゃくとり法

N,K = map(int, input().split())
sN = [int(input()) for i in range(N)]

# 初期値設定
ans,l,r = 0,0,1
tmp = sN[0]
if tmp <= K:
    ans += 1
# 左を動かすループ
while l < N:
    # 0があったら合計は絶対0(K以下)にできるので例外処理
    if sN[l] == 0:
        print(N)
        exit()
    # 右を動かすループ
    while r <= N and tmp <= K:
        # 最後だけ次がないので調整
        if r < N:
            # *次の数
            tmp *= sN[r]
        r += 1  
    ans = max(ans, r - l - 1)
    if l == r:
        r += 1
    # /前の数(1以下になる場合を考慮して切り上げ)
    tmp = -(-tmp // sN[l])
    l += 1
print(ans)