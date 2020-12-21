 # -*- coding: utf-8 -*-

# 入力を各変数に格納
N = int(input())

ans = 0
# 10の〇乗は10になるパターン
for i in range(1, 6):
	if N == 10 ** i:
		ans = 10
# 他は各桁を足した合計
if ans == 0:
	# 数値を文字列にして1文字ずつのリストにする
	strN_list = list(str(N))
	# ループして各桁を足した合計を計算
	for strN in strN_list:
		ans += int(strN)
print(ans)