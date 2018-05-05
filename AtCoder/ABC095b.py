# -*- coding: utf-8 -*-

# 種類の数と粉の総量
shurui_cnt, kona = map(int, input().split())
# 各種類の粉消費量
shurui_list = []
for i in range(shurui_cnt):
    shurui_list.append(int(input()))
# 作った数
total_cnt = 0

# まず全種類を一つずつ作る
for value in shurui_list:
    kona -= value
    total_cnt += 1

min_value = min(shurui_list)

# 残った粉で一番安いやつをなるだけ作る
while kona >= min_value:
    kona -= min_value
    total_cnt += 1

print(total_cnt)