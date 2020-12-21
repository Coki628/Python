# -*- coding: utf-8 -*-

# 入力を変数に格納
count = int(input())
num_list = [int(i) for i in input().split()]

result = 0
for i in range(count):
    # 合計初期化
    num_sum = 0
    # 開始位置を一つずつ前にずらす
    for j in range(i, count):
        num_sum += num_list[j]
        # 前から順番に足していって0になるやつを探す
        if num_sum == 0:
            result += 1

print(result)