# -*- coding: utf-8 -*-

# 人数
psn_cnt = int(input())

# 日数と最終日の余り
date_amari = input()
# 分割
da_list = date_amari.split()
date = int(da_list[0])
amari = int(da_list[1])

# 人数分の入力値を取得
psn_list = []
for i in range(psn_cnt):
	psn_list.append(int(input()))

choco_cnt_list = []
# 人数分のループ
for i in range(psn_cnt):
	# チョコ数の初期化
	choco_cnt = 0
	# 日数+1のループ
	for j in range(date + 1):
		# 条件に当てはまる数値が日数を超えていなければその人のチョコを+1
		if date >= psn_list[i] * j + 1:
#			print(str(i) + "add")
			choco_cnt += 1
		# 条件に当てはまる数値が日数を超えたところでその人のチョコ数をリストに詰めて次の人へ
		else:
			choco_cnt_list.append(choco_cnt)
#			print(str(i) + "added" + str(choco_cnt))
			break
# 全員のチョコ数を合計して余りも足す
choco_sum = 0
for choco_cnt in choco_cnt_list:
	choco_sum += choco_cnt
choco_sum += amari

print(choco_sum)