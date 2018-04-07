# -*- coding: utf-8 -*-

# 入力を各変数に格納
start_num, end_num, range_num = map(int, input().split())

result_list = []
for i in range(start_num, end_num + 1):
	# 前半の条件に当てはまるものを抽出
	if i < start_num + range_num:
		result_list.append(i)
	else:
		break

for i in range(end_num, start_num -1, -1):
	# 後半の条件に当てはまるものを抽出
	if i > end_num - range_num:
		result_list.append(i)
	else:
		break

# 重複削除
result_list = list(set(result_list))
# 昇順ソート
result_list.sort()

for num in result_list:
	print(num)
