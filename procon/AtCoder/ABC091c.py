# -*- coding: utf-8 -*-

# 青い点を探して処理する関数
def search_bluedot(start_x, start_y):
	# 更新する値をグローバル宣言
	global ans
	global field
	for i in range(start_x, dot_count*2):
		for j in range(start_y, dot_count*2):
			# 青い点を見つけたら、答えを+1して青い点を消す
			if field[i][j] == 2:
				ans += 1
				field[i][j] = 0
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
for i in range(dot_count*2):
	for j in range(dot_count*2):
		# 赤い点を見つけたら、そこより外側で青い点を探す
		if field[i][j] == 1:
			search_bluedot(i+1, j+1)

print(ans)

