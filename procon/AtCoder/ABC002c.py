# -*- coding: utf-8 -*-

# 入力の数値6つ
x, y, a, b, c, d = map(int, input().split())
# 計算
num1 = a*d - a*y - d*x - b*c + b*x + c*y
# 絶対値
num1 = abs(num1)
# 計算
num1 = num1 / 2
# 出力
print(num1)
