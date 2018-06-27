# -*- coding: utf-8 -*-

N, M = map(int, input().split())

# 各都市の道路数を集計するリスト
cities = [0 for i in range(N)]

# 道路の数ループで各都市の道路数を数えていく
for i in range(M):
    a, b = map(int, input().split())
    cities[a-1] += 1
    cities[b-1] += 1

# 集計した各都市の道路数を出力
for i in range(N):
    print(cities[i])