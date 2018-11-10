# -*- coding: utf-8 -*-

"""
人の参考にした版。きれい。
"""

N = int(input())

for i in range(N):
    A, B, C = map(int, input().split())

    if A % 2 != 0:
        # 10円1円合わせても100円ないのでアウト
        if B * 10 + C < 100:
            print('No')
            continue
        # うまく組み合わせて都合する
        if B < 10:
            C -= 100 - B * 10
            B = 0
        else:
            B -= 10
        
    if B % 2 != 0:
        if C < 10:
            print('No')
            continue
        C -= 10

    if C % 2 != 0:
        print('No')
        continue

    print('Yes')