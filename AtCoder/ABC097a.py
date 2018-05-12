# -*- coding: utf-8 -*-

# 入力を各変数に格納
a, b, c, d = map(int, input().split())

# 直接可能
if abs(a - c) <= d:
	print("Yes")
# 間接可能
elif abs(a - b) <= d and abs(b - c) <= d:
	print("Yes")
else:
	print("No")