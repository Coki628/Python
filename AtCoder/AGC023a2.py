# -*- coding: utf-8 -*-

import itertools

# 入力を変数に格納
count = int(input())
num_list = [int(i) for i in input().split()]

s = [0]
sum_tmp = 0
for num in num_list:
    sum_tmp += num
    s.append(sum_tmp)

pair_list = list(itertools.combinations(s, 2))

result = 0
for pair in pair_list:
    if pair[0] == pair[1]:
        result += 1

print(result)