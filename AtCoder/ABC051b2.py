# -*- coding: utf-8 -*-

K, S = map(int, input().split())

ans = 0
for x in range(K+1):
    for y in range(K+1):
        # zは全部回さないで、Sを満たすようなものだけ用意する
        z = S - x - y
        # 候補となるzがKの範囲内であれば有効
        if 0 <= z <= K:
            ans += 1
print(ans)