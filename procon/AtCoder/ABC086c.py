# -*- coding: utf-8 -*-

cnt = int(input())

for i in range(cnt):
    t, x, y = map(int, input().split())
    # 条件1：そもそも届かない
    if t < x + y:
        print("No")
        exit()
    # 条件2：奇数偶数が違うのでぴったり止まれない
    elif t % 2 != (x + y) % 2:
        print("No")
        exit()
print("Yes")