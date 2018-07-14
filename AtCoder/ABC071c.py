# -*- coding: utf-8 -*-

from collections import Counter 

N = int(input())
AN = list(map(int, input().split()))

# 各数値ごとに数を集計
AN_dict = Counter(AN)
AN_sorted = sorted(AN_dict.items())

hen1 = 0
hen2 = 0
# 低い値から見ていって順次最大に更新していく
for an in AN_sorted:
    # 正方形
    if an[1] >= 4:
        hen1 = an[0]
        hen2 = an[0]
    # 長方形
    elif an[1] >= 2:
        hen2 = hen1
        hen1 = an[0]      

print(hen1 * hen2)