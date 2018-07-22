# -*- coding: utf-8 -*-

# DFSでやる版

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

cnt = 0
p = [0] * N
def dfs(pos, mask):
    # 関数内で更新して外部で参照する変数のglobal宣言
    global cnt
    # 最後まで到達できた時はここを通る
    if pos == N:
        # 前提としてスタート地点(添字0)が0かどうか確認
        if p[0] == 0:
            for i in range(N-1):
                # 各辺が存在(各頂点ペアが隣接)しているか確認していく
                if graph[p[i]][p[i+1]] == False:
                    break
                # 全て存在すれば条件を満たすのでカウント+1
                if i == N-2:
                    cnt += 1
        return
    for i in range(N):
        # 残っている数の中にiがまだ残っている場合
        # (maskの中に(1<<i)のビットがまだ残っていればTrueを返す)
        if mask & (1<<i):
            p[pos] = i
            # 残っている数からiを取り除く
            # (排他的論理和XORで重複部分が0になる)
            dfs(pos+1, mask^(1<<i))
# 0からN-1までの数の集合
dfs(0, (1<<N)-1)
print(cnt)

# 参考：2進数の1を左へ3ビット移動(1→8)
# (1 << 3)