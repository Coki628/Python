# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

bN = []
if N % 2 == 0:
    for i in range(N):
        if i % 2 == 0:
            bN.append(str(aN[i]))
        else:
            bN.insert(0, str(aN[i]))
else:
    for i in range(N):
        if i % 2 == 1:
            bN.append(str(aN[i]))
        else:
            bN.insert(0, str(aN[i]))

print(" ".join(bN))