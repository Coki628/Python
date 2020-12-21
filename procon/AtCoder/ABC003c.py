# -*- coding: utf-8 -*-

N, K = map(int, input().split())
rN = list(map(int, input().split()))

rN.sort()
# 上位K人の動画のみにする
rN = rN[N-K:]

rate = 0
# その中で低い方から順に見る
for i in range(K):
    rate = (rate + rN[i]) / 2
print(rate)
