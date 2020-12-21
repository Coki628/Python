# -*- coding: utf-8 -*-

# 入力を各変数に格納(N,M,X)
cell_size, toll_size, start_cell = map(int, input().split())
# 数字が文字列リストなので数値リストに変換
toll_list = [int(i) for i in input().split()]

# 0からX
zero_to_start = 0
for i in range(0, start_cell):
	
	if i in toll_list:
		zero_to_start += 1

# XからN
start_to_end = 0
for i in range(start_cell, cell_size):

	if i in toll_list:
		start_to_end += 1

# 小さい方を出力
if zero_to_start <= start_to_end:
	print(zero_to_start)
else:
	print(start_to_end)
