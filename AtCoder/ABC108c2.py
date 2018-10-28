# -*- coding: utf-8 -*-

# mod Kをうまく使ってaを全検索

N,K = map(int, input().split())

num = [0] * K
# 元のNまでの数がmod Kの世界で何になるかを集計
for i in range(1,N+1):
    num[i%K] += 1

ans = 0
# a mod Kを決め打ちしてb mod Kとc mod Kを出す
for a in range(K):
    b = (K - a) % K
    c = (K - a) % K
    # つじつまが合ってるか確認
    if (b + c) % K == 0:
        # 予め集計してあるmod Kでの数を合計させる
        ans += num[a] * num[b] * num[c]
print(ans)