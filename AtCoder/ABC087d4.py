# -*- coding: utf-8 -*-

"""
参考：https://www.hamayanhamayan.com/entry/2018/01/28/231916
・幅優先
・辺の情報の持たせ方を工夫して、辺のループを余計に回さないようにする。
"""

from collections import deque

N, M = map(int, input().split())

l_edges = [[] for i in range(N)]
r_edges = [[] for i in range(N)]
for i in range(M):
    l, r, d = map(int, input().split())
    l_edges[l-1].append((r-1, d))
    r_edges[r-1].append((l-1, d))

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
        # 左から見た右側を確認
        for r, d in l_edges[cur]:
            # 訪問済なら矛盾がないか検証する
            if visited[r]:
                if pos[r] - pos[cur] != d:
                    print('No')
                    exit()
            # 未訪問なら今回の情報を記録する
            else:
                visited[r] = 1
                pos[r] = pos[cur] + d
                # 次回以降、開始位置として使うのでキューに詰める
                que.append(r)
        # 右から見た左側を確認
        for l, d in r_edges[cur]:
            if visited[l]:
                if pos[cur] - pos[l] != d:
                    print('No')
                    exit()
            else:
                visited[l] = 1
                pos[l] = pos[cur] - d
                que.append(l)

for i in range(N):
    # 訪問済(確認済)でなければ走破しに行く
    if not visited[i]:
        bfs(i)
# 最後まで矛盾がなければOK
print('Yes')
