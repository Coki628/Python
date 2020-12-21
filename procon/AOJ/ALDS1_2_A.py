# -*- coding: utf-8 -*-

"""
bubble sort
"""

N = int(input())
aN = list(map(str, input().split()))

done = False
cnt = 0
while not done:
    done = True
    for i in range(N-1, 0, -1):
        if int(aN[i]) < int(aN[i-1]):
            tmp = aN[i]
            aN[i] = aN[i-1]
            aN[i-1] = tmp
            done = False
            cnt += 1
print(' '.join(aN))
print(cnt)
