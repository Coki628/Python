# -*- coding: utf-8 -*-

"""
selection sort
"""

N = int(input())
aN = list(map(str, input().split()))

cnt = 0
for i in range(N):
    minj = i
    for j in range(i, N):
        if int(aN[j]) < int(aN[minj]):
            minj = j
    if i != minj:
        tmp = aN[i]
        aN[i] = aN[minj]
        aN[minj] = tmp
        cnt += 1
print(' '.join(aN))
print(cnt)
