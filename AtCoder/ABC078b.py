# -*- coding: utf-8 -*-

X, Y, Z = map(int, input().split())

for i in range(1, 1000001):
    if (Y + Z) * i + Z > X:
        # 座り切れなくなった所で、ひとつ前の値を出力
        print(i - 1)
        exit()