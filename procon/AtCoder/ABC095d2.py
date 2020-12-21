# -*- coding: utf-8 -*-

"""
今度は再帰で色々頑張ってみる…。
"""

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
memo[0] = 0
visited[0] = True
cnt = 0
def dfs(cur):
    global cnt
    if all(visited):
        return
    # 時計回り
    if cur != N:
        if not visited[cur+1]:
            if memo[cur+1] > memo[cur] + edges[cur][2] - sushi[cur+1]:
                memo[cur+1] = memo[cur] + edges[cur][2] - sushi[cur+1]
                visited[cur+1] = True
                dfs(cur+1)
                visited[cur+1] = False
        else:
            if memo[cur+1] > memo[cur] + edges[cur][2]:
                memo[cur+1] = memo[cur] + edges[cur][2]
                dfs(cur+1)
                visited[cur+1] = False
            
    else:
        if cnt >= 2:
            print(abs(min(memo)))
            exit()
        cnt += 1
        if memo[0] > memo[cur] + edges[0][2]:
            memo[0] = memo[cur] + edges[0][2]
            dfs(0)
    # 反時計回り
    if cur != 0:
        if not visited[cur-1]:
            if memo[cur-1] > memo[cur] + edges[cur-1][2] - sushi[cur-1]:
                memo[cur-1] = memo[cur] + edges[cur-1][2] - sushi[cur-1]
                visited[cur-1] = True
                dfs(cur-1)
                visited[cur-1] = False
        else:
            if memo[cur-1] > memo[cur] + edges[cur-1][2]:
                memo[cur-1] = memo[cur] + edges[cur-1][2]
                dfs(cur-1)
                visited[cur-1] = False

    else:
        if cnt >= 2:
            print(abs(min(memo)))
            exit()
        cnt += 1
        if not visited[N]:
            if memo[N] > memo[cur] + edges[N][2] - sushi[N]:
                memo[N] = memo[cur] + edges[N][2] - sushi[N]
                visited[N] = True
                dfs(N)
                visited[N] = False
        else:
            if memo[N] > memo[cur] + edges[N][2]:
                memo[N] = memo[cur] + edges[N][2]
                dfs(N)
                visited[N] = False

dfs(0)
print(abs(min(memo)))
