# -*- coding: utf-8 -*-

# 二分探索でやる版2(前の方は探す必要ないはず)
# 失敗。なんかTLEするわ… sliceするのが遅いのかな。

from bisect import bisect_left

N,M = map(int, input().split())
X,Y = map(int, input().split())
aN = list(map(int, input().split()))
bM = list(map(int, input().split()))

idx_a,idx_b,cnt = 0,0,0
time = aN[0]

while True:
    # A->Bがどこで出発できるか
    idx_a = bisect_left(aN[idx_a:], time) + idx_a
    # 終了判定
    if idx_a >= N:
        break
    # 現在時刻を進める
    time = aN[idx_a] + X

    # B->Aも同じようにやる
    idx_b = bisect_left(bM[idx_b:], time) + idx_b
    if idx_b >= M:
        break
    time = bM[idx_b] + Y

    # B->Aが行けたらカウント
    cnt += 1
print(cnt)