# -*- coding: utf-8 -*-

N = int(input())

for i in range(N):
    A, B, C = map(int, input().split())

    if A % 2 != 0:
        if B - 10 >= 0:
            B -= 10
        # 10円が10枚なかった
        else:
            while B < 10:
                if C - 10 >= 0:
                    B += 1
                    C -= 10
                # 10円10枚ないし1円も10枚なかった
                else:
                    print('No')
                    break
            else:
                # 内側ループが正常終了したらここですぐ下のcontinueをスキップ
                continue
            # breakしたらここで外側ループのcontinue
            continue

    if B % 2 != 0:
        if C - 10 >= 0:
            pass
        else:
            print('No')
            continue

    if C % 2 != 0:
        print('No')
        continue
        
    print('Yes')