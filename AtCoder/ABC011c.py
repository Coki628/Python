# -*- coding: utf-8 -*-

"""
メモ化再帰
"""

N = int(input())
NG = [0] * 3
NG[0] = int(input())
NG[1] = int(input())
NG[2] = int(input())
remain = 100

# ある残り回数(0～100)の時、ある地点(0～300)に来たかどうか
visited = [[False] * (remain+1) for i in range(301)]

def dfs(cur, remain):
    # 残り回数0
    if remain < 0:
        return
    # NGに当たった
    if cur in NG:
        return
    # 訪問済
    if visited[cur][remain]:
        return
    visited[cur][remain] = True
    # 無事到着
    if cur <= 0:
        print('YES')
        exit()
    # 1～3を引いていく
    for i in range(1, 4):
        dfs(cur-i, remain-1)

dfs(N, remain)
# 最後まで着かなかった
print('NO')
