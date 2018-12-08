# -*- coding: utf-8 -*-

from collections import Counter

N = int(input())
aN = [input() for i in range(N)]

# 名前ごとに数を集計
d = Counter(aN)
# 数の多い順でソート
l = d.most_common()

print(l[0][0])
