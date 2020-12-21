# -*- coding: utf-8 -*-

N, x = map(int, input().split())
aN = list(map(int, input().split()))

cnt = 0
# 初期値設定
if aN[0] > x:
    cnt += aN[0] - x
    aN[0] -= aN[0] - x

# 2番目以降を一つ前と比べていく
for i in range(1,N):
    if aN[i-1] + aN[i] > x:
        cnt += aN[i-1] + aN[i] - x
        aN[i] -= aN[i-1] + aN[i] - x

print(cnt)