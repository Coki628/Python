# -*- coding: utf-8 -*-

import itertools

N, M = map(int, input().split())

# 入力値の頂点ペア(辺)を格納する配列
a = [0] * M
b = [0] * M

# 頂点ペアが隣接しているかどうかを格納する配列
graph = [[False] * N for i in range(N)]

# 入力値の頂点ペア(辺)を格納していくループ
for i in range(M):
    a[i], b[i] = map(int, input().split())
    # 添字に合わせて入力値から1減らしておく
    a[i] -= 1
    b[i] -= 1
    # 入力値の内容に合わせて、隣接していることを記録
    graph[a[i]][b[i]] = True
    graph[b[i]][a[i]] = True

# 頂点をつなぐ全組み合わせ(順列)
# Nが少なければこれでいけるんだね…
p = list(itertools.permutations(range(N)))

cnt = 0
for i in range(len(p)):
    # 今回は出発点が1(添字0)と決まっているので、それ以外はbreak
    if p[i][0] != 0:
        break
    for j in range(len(p[i])-1):
        # 各辺が存在(各頂点ペアが隣接)しているか確認していく
        if graph[p[i][j]][p[i][j+1]] == False:
            break
        # 全て存在すれば条件を満たすのでカウント+1
        if j == len(p[i])-2:
            cnt += 1
print(cnt)