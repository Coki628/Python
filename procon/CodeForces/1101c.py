# -*- coding: utf-8 -*-

T = int(input())
for _ in range(T):
    N = int(input())
    x, y = map(int, input().split())
    mx_x1 = prv_x = x
    mx_y1 = prv_y = y
    ans = [1]
    for i in range(1, N):   
        x, y = map(int, input().split())
        if (x > prv_x and x > prv_y
                or prv_x > x and prv_x > y):
            pass
        prv_x = x
        prv_y = y
