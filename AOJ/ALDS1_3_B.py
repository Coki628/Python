# -*- coding: utf-8 -*-

"""
queue
"""

from collections import deque

N, Q = map(int, input().split())

que = deque()
for i in range(N):
    name, time = map(str, input().split())
    que.append([name, int(time)])

cnt = 0
while len(que) != 0:
    cur = que.popleft()
    # もう1回
    if cur[1] - Q > 0:
        cnt += Q
        que.append([cur[0], cur[1] - Q])
    # やりきった
    else:
        cnt += cur[1]
        print(cur[0], cnt)
