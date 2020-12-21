# -*- coding: utf-8 -*-

from collections import deque

T = int(input())
xT = [deque(input()) for i in range(T)]

for i in range(T):
    # 逆順と比べて回文判定
    if ''.join(xT[i]) == ''.join(reversed(xT[i])):
        # 全部いっこずらす
        tmp = xT[i].popleft()
        xT[i].append(tmp)
        # 回文じゃない判定
        if ''.join(xT[i]) != ''.join(reversed(xT[i])):
            print(''.join(xT[i]))
        else:
            print(-1)
    # 最初から回文じゃない
    else:
        print(''.join(xT[i]))
