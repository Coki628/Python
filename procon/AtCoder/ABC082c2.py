# -*- coding: utf-8 -*-

from collections import Counter 

cnt = int(input())
input_list = [int(i) for i in input().split()]

# 各数値ごとに数を集計(Counterオブジェクトで集計処理を簡略化)
num_dict = Counter(input_list)

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