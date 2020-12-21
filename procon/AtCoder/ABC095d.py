# -*- coding: utf-8 -*-

"""
幅優先
ダメだ…これだと途中で引き返して逆行った時、
訪問済フラグがおかしくなってる。
"""

from collections import deque

N, C = map(int, input().split())
x, v = map(int, input().split())
edges = [(0, 1, x)]
sushi = [0, v]
prev_x = x
for i in range(1, N):
    x, v = map(int, input().split())
    edges.append((i, i+1, x-prev_x))
    sushi.append(v)
    prev_x = x
edges.append((N, 0, C-prev_x))

# 最短距離メモ
memo = [float('inf')] * (N+1)
visited = [False] * (N+1)
que = deque()
# スタート位置
que.append(0)
memo[0] = 0
visited[0] = True
while len(que) != 0:
    cur = que.popleft()
    # 時計回り
    if cur != N:
        if memo[cur+1] > memo[cur] + edges[cur][2]:
            que.append(cur+1)
            memo[cur+1] = memo[cur] + edges[cur][2]
            if not visited[cur+1]:
                memo[cur+1] -= sushi[cur+1]
            visited[cur+1] = True
    else:
        if memo[0] > memo[cur] + edges[0][2]:
            que.append(0)
            memo[0] = memo[cur] + edges[0][2]
    # 反時計回り
    if cur != 0:
        if memo[cur-1] > memo[cur] + edges[cur-1][2]:
            que.append(cur-1)
            memo[cur-1] = memo[cur] + edges[cur-1][2]
            if not visited[cur-1]:
                memo[cur-1] -= sushi[cur-1]
            visited[cur-1] = True
    else:
        if memo[N] > memo[cur] + edges[N][2]:
            que.append(N)
            memo[N] = memo[cur] + edges[N][2]
            if not visited[N]:
                memo[N] -= sushi[N]
            visited[N] = True
print(abs(min(memo)))
