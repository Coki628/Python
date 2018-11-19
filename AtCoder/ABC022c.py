# -*- coding: utf-8 -*-

"""
地力で頑張ってDFS
→盛大にTLE
"""

import sys

# 再帰呼び出しの回数制限(デフォルト1000)
sys.setrecursionlimit(10 ** 9)

N,M = map(int, input().split())
uM,vM,lM = [0] * M, [0] * M, [0] * M
for i in range(M):
    uM[i], vM[i], lM[i] = map(int, input().split())

visited = [False] * M
ans = float('inf')

def dfs(cur, cnt):
    global ans
    # 始点に戻ってきた時
    if cur == 1 and cnt != 0:
        return min(cnt, ans)
    # 各道でループ
    for i in range(M):
        # 現在位置があるかと、逆が訪問済ではないかどうか
        if cur == uM[i] and not visited[i]:
            # 今回の場所を訪問済に
            visited[i] = True
            # 現在位置移動と移動距離追加して再帰やる
            ans = dfs(vM[i], cnt+lM[i])
            # 戻ってきたら戻す
            visited[i] = False
        # uとv逆なだけ
        if cur == vM[i] and not visited[i]:
            visited[i] = True
            ans = dfs(uM[i], cnt+lM[i])
            visited[i] = False
    # 向かう道が全部なくなったら返す
    return ans

ans = dfs(1, 0)
# 該当ルートがなかった時の例外処理
if ans == float('inf'):
    print(-1)
else:
    print(ans)
