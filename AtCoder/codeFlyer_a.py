 # -*- coding: utf-8 -*-

# 入力を各変数に格納
a = int(input())
b = int(input())

ans = 0

for i in range(b, 101, b):
	
	if i > a:
		break
	else:
		ans += b

print(ans)