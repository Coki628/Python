# -*- coding: utf-8 -*-

memo = [0] * (10 ** 5 + 1)
memo[2] = 1
add = 1
for i in range(3, len(memo)):
    memo[i] = memo[i-1] + add
    if i % 2 == 1:
        add += 1

N = int(input())

for i in range(2, len(memo)):
    if memo[i-1] + 1 <= N <= memo[i]:
        print(i)
        exit()
