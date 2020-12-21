# -*- coding: utf-8 -*-

cnt = int(input())
mochi_list = []
for i in range(cnt):
    mochi_list.append(int(input()))
# 餅リストから重複を削除
res_list = list(set(mochi_list))
# 結果リストの長さを出力
print(len(res_list))