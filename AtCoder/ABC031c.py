# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

taka = [[0] * N for i in range(N)]
aoki = [[0] * N for i in range(N)]

ans = float('-inf')
# takaのターン
for i in range(N):
    aoki_max = 0
    # aokiのターン
    for j in range(N):
        # 同じ場所にはできない
        if i == j:
            taka[i][j] = float('-inf')
            aoki[i][j] = float('-inf')
            continue
        # ルール通りに偶奇で得点配分
        for k in range(min(i,j),max(i+1,j+1)):
            # 開始位置によって偶奇は変わる
            if (k - min(i,j)) % 2:
                aoki[i][j] += aN[k]
            else:
                taka[i][j] += aN[k]
    # aokiが最大(同値は一番左)の時のtaka
    ans = max(taka[i][aoki[i].index(max(aoki[i]))], ans)

print(ans)