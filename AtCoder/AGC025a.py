 # -*- coding: utf-8 -*-

# 入力を各変数に格納
N = int(input())

ans = 0
# 10の〇乗は10になるパターン
if N == 10 ** 1 or N == 10 ** 2 or N == 10 ** 3 or N == 10 ** 4 or N == 10 ** 5:
	ans = 10
# 他は各桁を足した合計
else:
	# 数値を文字列にして1文字ずつのリストにする
	strN_list = list(str(N))
	# ループして各桁を足した合計を計算
	for strN in strN_list:
		ans += int(strN)
print(ans)