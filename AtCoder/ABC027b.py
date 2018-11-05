# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

# 均等にならない場合は-1
if sum(aN) % N != 0:
    print(-1)
else:
    target = sum(aN) // N
    tmp,start,cnt = 0,0,0
    for i in range(N):
        # 目標の人数と一致しない場所
        if aN[i] != target:
            # 開始位置
            if tmp == 0:
                start = i
            # 目標との差分を保持する
            tmp += aN[i] - target
            # 終了位置
            if tmp == 0:
                cnt += i - start
                start = 0
    print(cnt)