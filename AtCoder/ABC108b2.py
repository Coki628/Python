# -*- coding: utf-8 -*-

x1,y1,x2,y2 = map(int, input().split())

X = x2 - x1
Y = y2 - y1

# 3は2から、4は1から、同じ方向に動かす
print(str(x2 - Y) + ' ' + str(y2 + X) + ' ' + str(x1 - Y) + ' ' + str(y1 + X))