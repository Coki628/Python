# -*- coding: utf-8 -*-

# 入力の数値4つ
train1 = int(input())
train2 = int(input())
bus1 = int(input())
bus2 = int(input())

if train1 < train2:
	ans1 = train1
else:
	ans1 = train2

if bus1 < bus2:
	ans2 = bus1
else:
	ans2 = bus2

ans_sum = ans1 + ans2

print(ans_sum)