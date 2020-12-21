# -*- coding: utf-8 -*-

instr = input()
target = int(input())

# 部分文字列の一覧リストを作成
str_list = []
for i in range(len(instr)):

	for j in range(len(instr)+1):
		
		if i < j:
			str_list.append(instr[i:j])
		j += 1
	i += 1

# 重複削除
str_list = list(set(str_list))
# 昇順ソート
str_list.sort()

print(str_list[target-1])