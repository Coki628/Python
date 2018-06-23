# -*- coding: utf-8 -*-

N = int(input())

cnt = 1

while True:
    if cnt ** 2 > N:
        # オーバーしたらひとつ前の値を出力
        print((cnt -1) ** 2)
        exit()
    else:
        cnt += 1