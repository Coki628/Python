# -*- coding: utf-8 -*-

"""
参考：https://www.hamayanhamayan.com/entry/2018/01/28/231916
・幅優先
・でもこれだとまだO(N*M)だと思うからTLE前提
"""

from collections import deque

N, M = map(int, input().split())

edges = [None] * M
for i in range(M):
    l, r, d = map(int, input().split())
    edges[i] = (l-1, r-1, d)

# 訪問済メモ
visited = [0] * N

def bfs(start):
    # この開始位置を基準とした相対位置を記録する
    pos = {start: 0}
    que = deque()
    que.append(start)
    visited[start] = 1
    while len(que) != 0:
        cur = que.popleft()
        # 辺のループ
        for l, r, d in edges:
            # 左から見た右側を確認
            if l == cur:
                # 訪問済なら矛盾がないか検証する
                if visited[r]:
                    if pos[r] - pos[l] != d:
                        print('No')
                        exit()
                # 未訪問なら今回の情報を記録する
                else:
                    visited[r] = 1
                    pos[r] = pos[l] + d
                    # 次回以降、開始位置として使うのでキューに詰める
                    que.append(r)
            # 右から見た左側を確認
            elif r == cur:
                if visited[l]:
                    if pos[r] - pos[l] != d:
                        print('No')
                        exit()
                else:
                    visited[l] = 1
                    pos[l] = pos[r] - d
                    que.append(l)

for i in range(N):
    # 訪問済(確認済)でなければ走破しに行く
    if not visited[i]:
        bfs(i)
# 最後まで矛盾がなければOK
print('Yes')
