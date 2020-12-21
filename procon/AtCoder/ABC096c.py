# -*- coding: utf-8 -*-

# 入力を各変数に格納
x, y = map(int, input().split())

# キャンバスを2次元配列に格納
canvas = []
for i in range(x):
	# 文字列を1字ずつリストに入れる
	canvas.append(list(input()))

for i in range(x):
	for j in range(y):
		# 対象マスが#の時はチェックする
		if canvas[i][j] == "#":
			# 左端以外
			if i != 0:
				# 左隣
				if canvas[i-1][j] == "#":
					continue
			# 右端以外
			if i != x-1:
				# 右隣
				if canvas[i+1][j] == "#":
					continue
			# 最上段以外
			if j != 0:
				# ひとつ上
				if canvas[i][j-1] == "#":
					continue
			# 最下段以外
			if j != y-1:
				# ひとつ下
				if canvas[i][j+1] == "#":
					continue
			# どの条件も当てはまらない→四方に#がない
			print("No")
			exit()
# 一度もNoを通らなければYes
print("Yes")
