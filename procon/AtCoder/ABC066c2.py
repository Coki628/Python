# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

bN = [0] * N
if N % 2 == 0:
    for i in range(N):
        if i % 2 == 0:
            bN[(N//2) + i//2] = str(aN[i])
        else:
            bN[(N//2) - (i+1)//2] = str(aN[i])
else:
    for i in range(N):
        if i % 2 == 0:
            bN[(N//2) - i//2] = str(aN[i])
        else:
            bN[(N//2) + (i+1)//2] = str(aN[i])

print(" ".join(bN))