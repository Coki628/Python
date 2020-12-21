# -*- coding: utf-8 -*-

"""
参考：https://kimiyuki.net/writeup/algo/atcoder/arc-085-d/
　　　https://atcoder.jp/contests/abc078/submissions/3854338
・ゲーム木の探索(ミニマックス法)
・メモ化再帰
・メモの持ち方をシンプルにした版
　→無事AC。相手のスコアはメモらなくてもいいのか…。
"""

import sys

# 再帰呼び出しの回数制限(デフォルト1000)
sys.setrecursionlimit(10 ** 9)
 
N, Z, W = map(int, input().split())
aN = list(map(int, input().split()))

# メモ[手番][残り山札数] = このルートでのmax,min
memo = [[0] * (N+1) for _ in range(2)]

def dfs(turn, rest, x, y):
    if rest == 0:
        return abs(x - y)
    # 先手番
    if turn % 2 == 0:
        # 既に見た局面ならメモの内容を返す
        if memo[0][rest]: 
            return memo[0][rest]
        mx = -1
        for i in range(N-rest, N):
            x = aN[i]
            idx = i+1
            mx = max(dfs(turn+1, N-idx, x, y), mx)
        # ここより深い所を探し終わったので結果をメモする
        memo[0][rest] = mx
        return mx
    # 後手番
    else:
        if memo[1][rest]: 
            return memo[1][rest]
        mn = float('inf')
        for i in range(N-rest, N):
            y = aN[i]
            idx = i+1
            mn = min(dfs(turn+1, N-idx, x, y), mn)
        memo[1][rest] = mn
        return mn

print(dfs(0, N, Z, W))
