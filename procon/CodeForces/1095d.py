# -*- coding: utf-8 -*-

from collections import deque

N = int(input())
aN = []
for i in range(N):
    l = list(map(int, input().split()))
    aN.append(sorted(l))
aN.sort()

que = deque()
for i in range(1, N):
    if aN[i][0] == aN[i-1][0]:
        if not len(que):
            que.append(aN[i-1][1])
            que.append(aN[i-1][0])
            que.append(aN[i][1])
        elif aN[i-1][1] == que[0]:
            que.appendleft(aN[i][0])
            que.appendleft(aN[i][1])
        elif aN[i][1] == que[0]:
            que.appendleft(aN[i-1][0])
            que.appendleft(aN[i-1][1])
        elif aN[i-1][1] == que[-1]:
            que.append(aN[i][0])
            que.append(aN[i][1])
        elif aN[i][1] == que[-1]:
            que.append(aN[i-1][0])
            que.append(aN[i-1][1])
print(*que)
