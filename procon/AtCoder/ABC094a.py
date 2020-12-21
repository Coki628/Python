# -*- coding: utf-8 -*-

# 入力を各変数に格納
cat_num, both_num, sum_num = map(int, input().split())

if cat_num + both_num < sum_num or cat_num > sum_num:
	print("NO")
else:
	print("YES")
