# -*- coding: utf-8 -*-

# 赤い点を探して処理する関数
def search_reddot(end_x, end_y):
	# 更新する値をグローバル宣言
	global ans
	global field
	# Y軸向きには値の大きい方から優先して取るために、逆順でループ回す
	for y in range(end_y-1, -1, -1):
		for x in range(end_x):
			# 赤い点を見つけたら、答えを+1して赤い点を消す
			if field[y][x] == 1:
				ans += 1
				field[y][x] = 0
				return

dot_count = int(input())
# 点を置くフィールドを作成
field = [[0 for i in range(dot_count*2)] for j in range(dot_count*2)]
# 赤い点の配置
for i in range(dot_count):
	x, y = map(int, input().split())
	field[y][x] = 1
# 青い点の配置
for i in range(dot_count):
	x, y = map(int, input().split())
	field[y][x] = 2

ans = 0
for x in range(dot_count*2):
	# X軸向きに最も値の小さいものから取るために、Y軸ループを内側にする
	for y in range(dot_count*2):
		# 青い点を見つけたら、そこより内側で赤い点を探す
		if field[y][x] == 2:
			search_reddot(x, y)

print(ans)