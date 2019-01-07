# -*- coding: utf-8 -*-

"""
参考：https://www.hamayanhamayan.com/entry/2018/01/28/231916
・深さ優先
・幅優先とほとんど一緒だけど練習用にこっちも書いた。
"""

import sys

# 再帰呼び出しの回数制限(デフォルト1000)
sys.setrecursionlimit(10 ** 9)

N, M = map(int, input().split())

l_edges = [[] for i in range(N)]
r_edges = [[] for i in range(N)]
for i in range(M):
    l, r, d = map(int, input().split())
    l_edges[l-1].append((r-1, d))
    r_edges[r-1].append((l-1, d))

# 訪問済メモ
visited = [0] * N

def dfs(cur):
    visited[cur] = 1
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
            # 確認した位置を次の現在位置として見に行く
            dfs(r)
    # 右から見た左側を確認
    for l, d in r_edges[cur]:
        if visited[l]:
            if pos[cur] - pos[l] != d:
                print('No')
                exit()
        else:
            visited[l] = 1
            pos[l] = pos[cur] - d
            dfs(l)

for i in range(N):
    # 訪問済(確認済)でなければ走破しに行く
    if not visited[i]:
        # この開始位置を基準とした相対位置を記録する
        pos = {i: 0}
        dfs(i)
        pos = {}
# 最後まで矛盾がなければOK
print('Yes')
