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
for i in range(Q):
    for j in range(len(sN)):
        if tQ[i] == sN[j]:
            cnt += 1
            # 見つかったらiQ[i]についてこれ以上調べることない
            break
print(cnt)
