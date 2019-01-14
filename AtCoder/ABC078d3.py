# -*- coding: utf-8 -*-

"""
参考：https://kimiyuki.net/writeup/algo/atcoder/arc-085-d/
・ミニマックス法
・メモ化再帰
・先手と後手にそれぞれのメモを作る
・TLEしてpypyでAC(O(N^2)=2000^2=400万)
"""

import sys
from collections import defaultdict

# 再帰呼び出しの回数制限(デフォルト1000)
sys.setrecursionlimit(10 ** 9)
 
N, Z, W = map(int, input().split())
aN = list(map(int, input().split()))

# 先手番用メモ[残り山札数][相手のスコア] = このルートでのmax
# 後手番用メモ[残り山札数][相手のスコア] = このルートでのmin
memo1 = [defaultdict(int) for _ in range(N+1)]
memo2 = [defaultdict(int) for _ in range(N+1)]

def dfs(turn, rest, x, y):
    if rest == 0:
        return abs(x - y)
    # 先手番
    if turn % 2 == 0:
        # 既に見た局面ならメモの内容を返す
        if memo1[rest][y]: 
            return memo1[rest][y]
        mx = -1
        for i in range(N-rest, N):
            x = aN[i]
            idx = i+1
            mx = max(dfs(turn+1, N-idx, x, y), mx)
        # ここより深い所を探し終わったので結果をメモする
        memo1[rest][y] = mx
        return mx
    # 後手番
    else:
        if memo2[rest][x]: 
            return memo2[rest][x]
        mn = float('inf')
        for i in range(N-rest, N):
            y = aN[i]
            idx = i+1
            mn = min(dfs(turn+1, N-idx, x, y), mn)
        memo2[rest][x] = mn
        return mn

print(dfs(0, N, Z, W))
