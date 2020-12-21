# -*- coding: utf-8 -*-

# 入力を各変数に格納
limit = int(input())

b = 2
p = 2

# 1000までのべき乗をリストに格納
num_list = [1]
while b ** p <= 1000:

	while b ** p <= 1000:
		num_list.append(b ** p)
		p += 1

	b += 1
	p = 2

# 重複削除
num_list = list(set(num_list))
# 昇順ソート
num_list.sort()

# limit以下のものを結果リストに格納
result_list = []
for num in num_list:
	if num <= limit:
		result_list.append(num)
	else:
		break

# 最後の要素を出力
print(result_list[len(result_list)-1])