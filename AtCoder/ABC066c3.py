# -*- coding: utf-8 -*-

# 普通のリストじゃなくてこれ使えば、左への挿入も速いとのこと
from collections import deque

N = int(input())
aN = list(map(int, input().split()))

bN = deque()
if N % 2 == 0:
    for i in range(N):
        if i % 2 == 0:
            bN.append(str(aN[i]))
        else:
            bN.appendleft(str(aN[i]))
else:
    for i in range(N):
        if i % 2 == 1:
            bN.append(str(aN[i]))
        else:
            bN.appendleft(str(aN[i]))

print(" ".join(bN))