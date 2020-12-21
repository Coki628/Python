# -*- coding: utf-8 -*-

from collections import Counter 

N = int(input())
AN = list(map(int, input().split()))

# 各数値ごとに数を集計
AN_dict = Counter(AN)

ans = 0
for key, value in AN_dict.items():
    tmp = 0
    # その数、+1、-1した数の合計を調べて、最大になるものを選ぶ
    tmp += value
    if key-1 in AN_dict:
        tmp += AN_dict[key-1]
    if key+1 in AN_dict:
        tmp += AN_dict[key+1]
    ans = max(ans, tmp)

print(ans)