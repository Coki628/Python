# -*- coding: utf-8 -*-

T = int(input())
xT = [int(input()) for i in range(T)]

for i in range(T):
    print(-(-xT[i] // 7))

