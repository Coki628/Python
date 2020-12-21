# -*- coding: utf-8 -*-

# 二分探索でやる版

from bisect import bisect_left

N,M = map(int, input().split())
X,Y = map(int, input().split())
aN = list(map(int, input().split()))
bM = list(map(int, input().split()))

idx_a,idx_b,cnt = 0,0,0
time = aN[0]

while True:
    # A->Bがどこで出発できるか
    idx_a = bisect_left(aN, time)
    # 終了判定
    if idx_a == N:
        break
    # 現在時刻を進める
    time = aN[idx_a] + X

    # B->Aも同じようにやる
    idx_b = bisect_left(bM, time)
    if idx_b == M:
        break
    time = bM[idx_b] + Y

    # B->Aが行けたらカウント
    cnt += 1
print(cnt)