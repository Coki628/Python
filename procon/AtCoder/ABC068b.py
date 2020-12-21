# -*- coding: utf-8 -*-

N = int(input())

cnt = 1
while True:
    if 2 ** cnt > N:
        print(2 ** (cnt - 1))
        exit()
    cnt += 1