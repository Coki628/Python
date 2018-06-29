# -*- coding: utf-8 -*-

N, M = map(int, input().split())
N_list = [list(map(int, input().split())) for i in range(N)]
M_list = [list(map(int, input().split())) for i in range(M)]

# 学生の数ループ
for i in range(N):
    # 最小値の初期値は無限大
    min_point = float("inf")
    ans = 0
    # チェックポイントの数ループ
    for j in range(M):
        # マンハッタン距離の計算
        x = abs(N_list[i][0] - M_list[j][0])
        y = abs(N_list[i][1] - M_list[j][1])
        tmp = x + y
        # 最短距離なら、最短距離を更新してチェックポイントの番号を出力
        if tmp < min_point:
            min_point = tmp
            ans = j + 1
    print(ans)