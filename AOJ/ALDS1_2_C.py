# -*- coding: utf-8 -*-

"""
stable sort
"""

from copy import deepcopy

N = int(input())
cN = list(map(str, input().split()))
cN2 = deepcopy(cN)

# bubble sort
for i in range(N):
    for j in range(N-1, i, -1):
        if int(cN[j][1]) < int(cN[j-1][1]):
            tmp = cN[j]
            cN[j] = cN[j-1]
            cN[j-1] = tmp
print(' '.join(cN))
print('Stable')

# selection sort
for i in range(N):
    minj = i
    for j in range(i, N):
        if int(cN2[j][1]) < int(cN2[minj][1]):
            minj = j
    tmp = cN2[i]
    cN2[i] = cN2[minj]
    cN2[minj] = tmp
print(' '.join(cN2))

for i in range(N):
    if cN[i] != cN2[i]:
        print('Not stable')
        exit()
print('Stable')
