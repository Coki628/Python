# -*- coding: utf-8 -*-

from decimal import Decimal, ROUND_HALF_UP

# 入力の数値
deg, dis = map(int, input().split())

# 風向の取得
def getDirection(deg):
	deg /= 10
	# 風向をリストに格納
	direction_list = ["NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S",
	 "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
	# 角度が以下の範囲の時はNで確定
	if deg < 11.25 or deg >= 348.75:
		return "N"
	# それ以外は22.5度ずつ増やしながら比較していく
	for i in range(len(direction_list)):
		a = 33.75 + i * 22.5
		if deg < a:
			return direction_list[i]

direction = getDirection(deg)

# 風程を風速にする
speed = dis / 60
# 小数第2位を四捨五入(python3はroundだけだといけない)
num1 = Decimal(str(speed))
num1 = num1.quantize(Decimal(".1"), ROUND_HALF_UP)
speed = float(num1)

# 風力の取得
def getWindPower(speed):
	# 風速をリストに格納
	speed_list = [0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]
	# 風速が以下の時は風力12で確定
	if speed >= 32.7:
		return "12"
	# それ以外はリストの数値を使って比較していく
	for i in range(len(speed_list)):
		# 最初だけ規則外なので別処理
		if i == 0:
			min_speed = 0.0
		# その後は規則的なのでカウンターを活用する
		else:
			min_speed = speed_list[i - 1]
		# 丸めないと0.3-0.1が0.19999とかなる
		max_speed = round(speed_list[i] - 0.1, 1)
		if min_speed <= speed <= max_speed:
			return str(i)

wind_power = getWindPower(speed)

# 風力が0なら風向はCで確定
if wind_power == "0":
	direction = "C"

print(direction + " " + wind_power)