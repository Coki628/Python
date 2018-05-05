# -*- coding: utf-8 -*-

# 入力を各変数に格納
num_list = [int(i) for i in input().split()]
times = int(input())

# 3つの数値を昇順ソート
num_list.sort()
# リスト内最大値を一時変数へ
num_tmp = num_list[2]
# 操作回数分の*2をする
for i in range(times):
	num_tmp = num_tmp * 2
num_list[2] = num_tmp

# リスト内の数値を合計して出力
print(sum(num_list))