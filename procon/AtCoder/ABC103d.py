# -*- coding: utf-8 -*-

N,M = map(int, input().split())
abM = [list(map(int, input().split())) for i in range(M)]

# 要望を西側の島でソートしておく
abM.sort()
# 初期値設定
tmp_e = abM[0][1]
cnt = 1
# 東島の位置を最小にしながら、次の西島と比較していく
for i in range(1,M):
    if tmp_e > abM[i][0]:
        tmp_e = min(abM[i][1], tmp_e)
    else:
        cnt += 1
        tmp_e = abM[i][1]
print(cnt)