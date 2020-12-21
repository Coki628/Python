# -*- coding: utf-8 -*-

A, B, C, D, E, F = map(int, input().split())

water = []
for i in range(A*100, F+1, A*100):
    water.append(i)
for i in range(B*100, F+1, B*100):
    water.append(i)
water = sorted(list(set(water)), reverse=True)

suger = []
for i in range(0, F+1, C):
    suger.append(i)
for i in range(0, F+1, D):
    suger.append(i)
suger = sorted(list(set(suger)))

ans_ws = 0
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