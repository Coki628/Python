# -*- coding: utf-8 -*-

# 入力を各変数に格納
month, date = map(int, input().split())

if date >= month:
	print(month)
else:
	print(month - 1)
