# -*- coding: utf-8 -*-

"""
自分のやつ手直しした版。きたねえ。
"""

N = int(input())

for i in range(N):
    end_flag = False
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
                    end_flag = True
                    break
            B = 0
            # breakしたらここで外側ループのcontinue
            if end_flag:
                continue

    if B % 2 != 0:
        if C - 10 >= 0:
            C -= 10
        else:
            print('No')
            continue

    if C % 2 != 0:
        print('No')
        continue
        
    print('Yes')