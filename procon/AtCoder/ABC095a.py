# -*- coding: utf-8 -*-

# 入力を変数に格納
toppings = input()

cost = 700

cnt = toppings.count("o")

cost += (100 * cnt)

print(cost)