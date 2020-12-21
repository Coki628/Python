# -*- coding: utf-8 -*-

L,H = map(int, input().split())
N = int(input())

for i in range(N):
    a = int(input())
    if a < L:
        print(L - a)
    elif L <= a <= H:
        print(0)
    elif a > H:
        print(-1)