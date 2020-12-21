# -*- coding: utf-8 -*-

"""
insertion sort
"""

N = int(input())
aN = list(map(str, input().split()))

for i in range(N):
    tmp = aN[i]
    j = i - 1
    while j >= 0 and int(aN[j]) > int(tmp):
        aN[j+1] = aN[j]
        j -= 1
    aN[j+1] = tmp

    print(' '.join(aN))
