# -*- coding: utf-8 -*-

N = int(input())
aN = [int(input()) for i in range(N)]

# 現在位置を保持する変数を使う
now = 1
for i in range(N):
    now = aN[now-1]
    if now == 2:
        print(i + 1)
        exit()
print(-1)