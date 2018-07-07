# -*- coding: utf-8 -*-

a, b = map(int, input().split())

# 本来の各柱の高さリスト
columns = [(i ** 2 + i) // 2 for i in range(1, 1000)]

for i in range(998):
    ab = columns[i+1] - columns[i]
    # 本来の柱と今見えている柱で差が一致する所を探す
    if ab == b - a:
        print(columns[i] - a)
        exit()