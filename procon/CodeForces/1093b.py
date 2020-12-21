# -*- coding: utf-8 -*-

T = int(input())
xT = [list(input()) for i in range(T)]

for i in range(T):
    # 逆順と比べて回文判定
    if ''.join(xT[i]) == ''.join(xT[i][::-1]):
        for j in range(len(xT[i])-1):
            # ひとつ先と位置交換
            xT[i][j], xT[i][j+1] = xT[i][j+1], xT[i][j]
            # 回文じゃない判定
            if ''.join(xT[i]) != ''.join(xT[i][::-1]):
                print(''.join(xT[i]))
                break
        # 最後まで回文だった
        else:
            print(-1)
    # 最初から回文じゃない
    else:
        print(''.join(xT[i]))
