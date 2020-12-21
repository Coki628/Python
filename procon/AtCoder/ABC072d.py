# -*- coding: utf-8 -*-

"""
・すんなり自力AC
"""

N = int(input())
pN = list(map(int, input().split()))

same = [False] * N
for i in range(N):
    if i+1 == pN[i]:
        same[i] = True
# 終端処理用に追加
same.append(False)

cnt = 0
for i in range(1, N+1):
    if same[i] and same[i-1]:
        same[i] = False
        same[i-1] = False
        cnt += 1
    elif not same[i] and same[i-1]:
        same[i-1] = False
        cnt += 1
print(cnt)
