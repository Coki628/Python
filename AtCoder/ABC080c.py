# -*- coding: utf-8 -*-

import itertools

N = int(input())
F = []
for i in range(N):
    F.append(list(map(int, input().split())))
P = []
for i in range(N):
    P.append(list(map(int, input().split())))

# 重複順列で自分の店の全パターンを出す
me_list = list(itertools.product([0, 1], repeat=10))

# 初期値は利益の最小値
me_p = -10000000
# 自分の店の全パターンループ
for i in range(len(me_list)):

    # 一度も営業しないパターンを除く
    if i == 0:
        continue

    tmp_p = 0
    # 他店の数でループ
    for j in range(N):

        cnt = 0
        # 月～金のAMPMループ
        for k in range(10):

            # 他店との営業重複カウント
            if me_list[i][k] == 1 and F[j][k] == 1:
                cnt += 1
        
        # 各店舗との重複数に応じた利益を合算していく
        tmp_p += P[j][cnt]

    # 合算した利益を現状の最高値と比較して置き換えていく
    me_p = max(me_p, tmp_p)   

print(me_p)