# -*- coding: utf-8 -*-

"""
二分探索
"""

from bisect import bisect_left

N = int(input())
sN = list(map(int, input().split()))
Q = int(input())
tQ = list(map(int, input().split()))

cnt = 0
for i in range(Q):
    idx = bisect_left(sN, tQ[i])
    # 探す要素が大きいとidxが範囲外にでるのでidx < N
    if idx < N and sN[idx] == tQ[i]:
        cnt += 1
print(cnt)
