# -*- coding: utf-8 -*-

from collections import Counter 

N = int(input())
AN = [int(input()) for i in range(N)]

# 各数値ごとに数を集計
AN_dict = Counter(AN)

cnt = 0
for key, value in AN_dict.items():
    if value % 2 != 0:
        cnt += 1
print(cnt)