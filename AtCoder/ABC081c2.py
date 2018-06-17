# -*- coding: utf-8 -*-

from collections import Counter 

N, K = map(int, input().split())
num_list = list(map(int, input().split()))

# 各数値ごとに数を集計
num_dict = Counter(num_list)
# 数の多い順でソート
num_tpl = num_dict.most_common()

# 種類が少ない時は書き換え0回
if len(num_tpl) <= K:
    print(0)
    exit()
else:
    sum1 = 0
    # 残していい数を集計
    for i in range(K):
        sum1 += num_tpl[i][1]
    # 全体の数 - 残す数で、書き換える数を出す
    print(N - sum1)