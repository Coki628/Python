# -*- coding: utf-8 -*-

"""
線形探索
"""

N = int(input())
sN = list(map(int, input().split()))
Q = int(input())
tQ = list(map(int, input().split()))

sN = list(set(sN))

cnt = 0
for i in range(len(sN)):
    for j in range(Q):
        if tQ[j] == sN[i]:
            cnt += 1
            
print(cnt)
