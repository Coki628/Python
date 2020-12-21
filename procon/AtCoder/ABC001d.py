# -*- coding: utf-8 -*-

import datetime
from operator import itemgetter

# 入力の数値
counter = int(input())
input_list = [input() for _ in range(counter)]

# 開始時刻と終了時刻のリスト
from_list = []
to_list = []
for input_str in input_list:
	from_list.append(input_str[0:4])
	to_list.append(input_str[5:9])

print(from_list)
print(to_list)

# 切り下げる時
def roundDownTime(time_str):
	# 0～4は0に
	if 0 <= int(time_str[3:4]) <= 4:
		return time_str[0:3] + "0"
	# 5～9は5に
	if 5 <= int(time_str[3:4]) <= 9:
		return time_str[0:3] + "5"

# 切り上げる時
def roundUpTime(time_str):
	# 0～4は5に
	if 0 <= int(time_str[3:4]) <= 4:
		return time_str[0:3] + "5"
	# 5～9は0に
	if 5 <= int(time_str[3:4]) <= 9:
		# 50分台だけは時のケタが動く
		if time_str[2:3] == "5":
			return str(int(time_str[0:2]) + 1) + "00"
		# 通常は10分のケタが動く
		else:
			return time_str[0:2] + str(int(time_str[2:3]) + 1) + "0"

# 丸め処理の完了したリスト
round_from_list = []
round_to_list = []
for i in range(counter):
	round_from_list.append(roundDownTime(from_list[i]))
	round_to_list.append(roundUpTime(to_list[i]))

print(round_from_list)
print(round_to_list)

# 開始時刻と終了時刻のペアで二次元リストを作る
time_list = [[0 for col in range(2)] for row in range(counter)]
for i in range(counter):
	time_list[i][0] = round_from_list[i]
	time_list[i][1] = round_to_list[i]
print(time_list)
# 開始時刻で昇順ソート
time_list.sort(key=itemgetter(0))
print(time_list)

#	from_list.append(datetime.time(int(input_str[0:2]), int(input_str[2:4]), 0))
#	to_list.append(datetime.time(int(input_str[5:7]), int(input_str[7:9]), 0))

