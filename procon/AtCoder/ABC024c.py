# -*- coding: utf-8 -*-

N,D,K = map(int, input().split())
lrD = [list(map(int, input().split())) for i in range(D)]
stK = [list(map(int, input().split())) for i in range(K)]

ans = [0] * K
for i in range(D):
    for j in range(K):
        # 左右どっちに動くか
        if stK[j][0] < stK[j][1]:
            # 始点sが範囲内かどうか
            if stK[j][0] >= lrD[i][0]:
                # なるべく右に移動
                stK[j][0] = max(lrD[i][1], stK[j][0])
                # 目的地より先まで行ければ確定
                if stK[j][0] >= stK[j][1]:
                    ans[j] = i + 1
                    # これ以降のループで更新しないように、始点=終点として到着済みにする
                    stK[j][0] = stK[j][1]
        elif stK[j][0] > stK[j][1]:
            if stK[j][0] <= lrD[i][1]:
                # なるべく左に移動
                stK[j][0] = min(lrD[i][0], stK[j][0])
                if stK[j][0] <= stK[j][1]:
                    ans[j] = i + 1
                    stK[j][0] = stK[j][1]

for i in range(K):
    print(ans[i])