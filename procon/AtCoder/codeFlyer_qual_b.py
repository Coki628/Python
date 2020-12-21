 # -*- coding: utf-8 -*-

# 入力を各変数に格納
shorty, cheezy, cust = map(int, input().split())
str_list = list(input())

# 人数分(文字数分)ループ
for x in str_list:
	# Sの場合
	if x == "S":
		if shorty != 0:
			shorty -= 1
	# Cの場合
	elif x == "C":
		if cheezy != 0:
			cheezy -= 1
	# Eの場合
	elif x == "E":
		# 両方売り切れなら何もしない
		if shorty == 0 and cheezy == 0:
			continue
		# ショートケーキを買う
		elif shorty >= cheezy:
			shorty -= 1
		# チーズケーキを買う
		elif cheezy > shorty:
			cheezy -= 1

print(shorty)
print(cheezy)