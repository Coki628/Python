# -*- coding: utf-8 -*-

cnt = int(input())
num_list = list(map(int, input().split()))

ans = 0
while True:
    for i in range(cnt):
        # 各数値に対して偶数か判定、奇数があればそこで終了
        if num_list[i] % 2 != 0:
            print(ans)
            exit()
        # 偶数なら操作を行う
        num_list[i] /= 2
    # 全ての数値で操作が完了したら+1
    ans += 1