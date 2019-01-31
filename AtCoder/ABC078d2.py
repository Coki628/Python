# -*- coding: utf-8 -*-

"""
・ゲーム木の探索(ミニマックス法)
・メモ化再帰
・頑張って作ったけどWA。しかも想定解は思いつくか勝負の一発解法…。
・この後メモの内容と置き方だけ直したらACした。惜しかったー。
"""

import sys

# 再帰呼び出しの回数制限(デフォルト1000)
sys.setrecursionlimit(10 ** 9)
 
N, Z, W = map(int, input().split())
aN = list(map(int, input().split()))

# メモ[手番][残り山札数]{現在のスコア}
memo = [[set() for _ in range(N+1)] for _ in range(2)]

def dfs(turn, rest, x, y):
    if rest == 0:
        return abs(x - y)
    if turn % 2 == 0:
        # 訪問済にする
        memo[turn%2][rest].add(abs(x - y))
        mx = -1
        for i in range(N-rest, N):
            x = aN[i]
            idx = i+1
            # 既に見た局面でなければ、再帰処理を続ける
            if abs(x - y) not in memo[1-turn%2][N-idx]:
                mx = max(dfs(turn+1, N-idx, x, y), mx)
        return mx
    else:
        memo[turn%2][rest].add(abs(x - y))
        mn = float('inf')
        for i in range(N-rest, N):
            y = aN[i]
            idx = i+1
            if abs(x - y) not in memo[1-turn%2][N-idx]:
                mn = min(dfs(turn+1, N-idx, x, y), mn)
        return mn

print(dfs(0, N, Z, W))
