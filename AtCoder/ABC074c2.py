# -*- coding: utf-8 -*-

A, B, C, D, E, F = map(int, input().split())

water = []
for i in range(0, F+1, A*100):
    for j in range(0, F+1, B*100):
        if i + j <= F:
            water.append(i + j)
water = sorted(list(set(water)), reverse=True)
# 0はあとで割る時に困るので外す
water.pop(len(water)-1)

suger = []
for i in range(0, F+1, C):
    for j in range(0, F+1, D):
        if i + j <= F:
            suger.append(i + j)
suger = sorted(list(set(suger)))

# 合計の初期値に水を入れておく(これは0でもいい気がするんだけど…)
ans_ws = water[0]
ans_s = 0
node = 0
# 水は多い方、砂糖は少ない方から見ていく
for i in range(len(water)):
    for j in range(len(suger)):
        # 合計量チェック
        if water[i] + suger[j] > F:
            break
        # 溶ける量チェック
        if suger[j] / (water[i] + suger[j]) <= E / (E + 100):
            # 最大濃度更新なら答えを更新
            if suger[j] / (water[i] + suger[j]) > node:
                node = suger[j] / (water[i] + suger[j])
                ans_ws = water[i] + suger[j]
                ans_s = suger[j]
print(str(ans_ws) + " " + str(ans_s))