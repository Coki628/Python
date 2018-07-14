# -*- coding: utf-8 -*-

N, M = map(int, input().split())
abM = [list(map(int, input().split())) for i in range(M)]

# 母体の削減を試みてみる
abM2 = []
for i in range(M):
    if abM[i][0] == 1 or abM[i][1] == N:
        abM2.append(abM[i])

transit = []
for i in range(len(abM2)):
    if abM2[i][0] == 1:
        transit.append(abM2[i][1])

for i in range(len(abM2)):
    if abM2[i][1] == N:
        if abM2[i][0] in transit:
            print("POSSIBLE")
            exit()
        
print("IMPOSSIBLE")