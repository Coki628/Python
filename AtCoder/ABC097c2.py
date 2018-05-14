# -*- coding: utf-8 -*-

instr = input()
target = int(input())

# 部分文字列の一覧リストを作成
str_list = []
for i in range(len(instr)):

	# 開始位置はi以降(i > jはやる意味ない)
	for j in range(i, len(instr)+1):
		
		# targetよりも大きい文字数のものは処理しない
		if j-i > target:
			break

		# 候補となる部分文字列の格納
		if i < j:
			str_list.append(instr[i:j])

		j += 1
	i += 1

# 重複削除
str_list = list(set(str_list))
# 昇順ソート
str_list.sort()

print(str_list[target-1])