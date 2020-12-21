# -*- coding: utf-8 -*-

"""
メモ化再帰
(DFS内で出力やんないで真偽返す版 + 条件整理)
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
    # 残り回数0 or NGに当たった or 訪問済
    if remain < 0 or cur in NG or visited[cur][remain]:
        return False
    visited[cur][remain] = True
    # 無事到着
    if cur <= 0:
        return True
    # 1～3を引いていく
    for i in range(1, 4):
        if(dfs(cur-i, remain-1)):
            # 着いたら即返却
            return True
    return False

if(dfs(N, remain)):
    print('YES')
# 最後まで着かなかった
else:
    print('NO')
