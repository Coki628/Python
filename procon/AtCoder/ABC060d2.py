# -*- coding: utf-8 -*-

"""
・メモ化再帰
・自力AC
・wの値は大きいけど限定的なので、添字にdict使えばOK
・ナップザックを普通のDPじゃなくてメモ化再帰でやってみた。
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)

from collections import defaultdict

N, W = map(int, input().split())
wN = [0] * N
vN = [0] * N
for i in range(N):
    wN[i], vN[i] = map(int, input().split())

# memo[i][j]: i番目以降の品物から重さの和がj以下なるように選んだときの価値の和の最大値
memo = [defaultdict(lambda: -1) for i in range(N+1)]

def dfs(cur, wei):
    # 既に調べた場所はそのまま返却
    if memo[cur][wei] != -1:
        return memo[cur][wei]
    # 品物を最後まで見た
    if cur == N:
        res = 0
    # まだ余力があるから両方試す
    elif wei >= wN[cur]:
        res = max(dfs(cur+1, wei-wN[cur]) + vN[cur], dfs(cur+1, wei))
    # もうこの回の荷物は入らない
    else:
        res = dfs(cur+1, wei)
    # ここまでのベストな値をメモに記録してから返却
    memo[cur][wei] = res
    return res
# 0番目以降で容量W以下の場合の結果を表示
print(dfs(0, W))
