# -*- coding: utf-8 -*-

from collections import Counter

N,K = map(int, input().split())
aN = list(map(int, input().split()))

# 各数値ごとに数を集計
counter = Counter(aN)
# 数の多い順でソート
counter = counter.most_common()
# 一番多かった道具の数
max_utensils = counter[0][1]
# 1人当たり料理数
dishes = -(-max_utensils // K)
# あったはずの1人当たり道具数
exp_utensils = dishes * len(list(set(aN)))

print(exp_utensils*K - N)
