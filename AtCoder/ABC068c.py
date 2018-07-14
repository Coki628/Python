# -*- coding: utf-8 -*-

N, M = map(int, input().split())
abM = [list(map(int, input().split())) for i in range(M)]

transit = []
for i in range(M):
    if abM[i][0] == 1:
        transit.append(abM[i][1])

for i in range(M):
    if abM[i][0] in transit and abM[i][1] == N:
        print("POSSIBLE")
        exit()
        
print("IMPOSSIBLE")