# -*- coding: utf-8 -*-

N = int(input())
colors = [0] * 1000002

# いもすやる
for i in range(N):
    a, b = map(int, input().split())
    colors[a] += 1
    colors[b+1] -= 1

# 合計する(累積和)
total = [0] * len(colors)
total[0] = colors[0]
for i in range(1, len(colors)):
    total[i] += colors[i] + total[i-1]

# 一番多い人数
print(max(total))
