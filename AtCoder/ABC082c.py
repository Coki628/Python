# -*- coding: utf-8 -*-

cnt = int(input())
input_list = [int(i) for i in input().split()]

num_dict = {}
# 各数値ごとに数を集計
for num in input_list:
    # 辞書にその数値のキーが存在すれば+1追加
    if num in num_dict:
        num_dict[num] += 1
    # なければ新たにキー作成
    else:
        num_dict[num] = 1

ans = 0
# 数値(key)と個数(value)の大小によって処理を分ける
for key, value in num_dict.items():
    # valueのが大きければkeyとvalueを揃える
    if key < value:
        ans += value - key
    # 元々揃っていれば何もしない
    elif key == value:
        continue
    # keyのが大きければその値を取り除く
    elif key > value:
        ans += value
print(ans)