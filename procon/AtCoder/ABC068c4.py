# -*- coding: utf-8 -*-

N, M = map(int, input().split())

# 1からある島への便があるかを保持
transit = [False] * N
# ある島からNへの便があるかを保持
transit2 = [False] * N

for i in range(M):
    a, b = map(int, input().split())
    if a == 1:
        transit[b] = True
    if b == N:
        transit2[a] = True

for i in range(N):
    if transit[i] and transit2[i]:
        print("POSSIBLE")
        exit()
print("IMPOSSIBLE")