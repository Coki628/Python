# -*- coding: utf-8 -*-

list_size = int(input())
num_list = [int(i) for i in input().split()]
# 昇順ソート
num_list2 = sorted(num_list)
# 中央値候補
center_left = num_list2[int(list_size / 2) - 1]
center_right = num_list2[int(list_size / 2)]

for i in range(list_size):
	# 除く値が左寄り(小さい)なら右、右寄り(大きい)なら左
	if num_list[i] <= center_left:
		print(center_right)
	else:
		print(center_left)