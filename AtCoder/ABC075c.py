# -*- coding: utf-8 -*-

N, M = map(int, input().split())

# 入力値の頂点ペア(辺)を格納する配列
a = [0] * 50
b = [0] * 50
# 頂点ペアが隣接しているかどうかを格納する配列
graph = [[False for i in range(50)] for j in range(50)]
# ある頂点が訪問済かどうかを格納する配列
visited = [False] * N

# 入力値の頂点ペア(辺)を格納していくループ
for i in range(M):
    a[i], b[i] = map(int, input().split())
    # 添字に合わせて入力値から1減らしておく
    a[i] -= 1
    b[i] -= 1
    # 入力値の内容に合わせて、隣接していることを記録
    graph[a[i]][b[i]] = True
    graph[b[i]][a[i]] = True

# 連結かどうか(全ての頂点が訪問済かどうか)を判定するDFS
def dfs(v):
    # 関数内で使用する変数のglobal宣言
    global graph
    global visited
    # 通った頂点は訪問済にする
    visited[v] = True
    for v2 in range(N):
        # 隣接していないor訪問済はやる意味ないのでcontinue
        if graph[v][v2] == False or visited[v2] == True:
            continue
        # 調べる必要のある箇所は再帰的に調べていく
        dfs(v2)

ans = 0
# 各辺について、橋かどうかを判定していくループ
for i in range(M):
    # まず該当の辺を外す(この状態で連結ではなくなってしまえばそれは橋)
    graph[a[i]][b[i]] = False
    graph[b[i]][a[i]] = False
    # 訪問済判定を初期化
    for j in range(N):
        visited[j] = False
    # 連結かどうか(全ての頂点が訪問済かどうか)を判定するDFS
    dfs(0)
    # 各頂点について、訪問済かどうかを確認していく
    bridge = False
    for j in range(N):
        # 頂点で一つでも未訪問の箇所があれば橋と判定する
        if visited[j] == False:
            bridge = True
    if bridge:
        ans += 1
    # 外した辺を元に戻す
    graph[a[i]][b[i]] = True
    graph[b[i]][a[i]] = True

print(ans)