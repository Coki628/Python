# -*- coding: utf-8 -*-

"""
参考：https://tdpc.contest.atcoder.jp/submissions/2750904
　　　https://tdpc.contest.atcoder.jp/submissions/3822099
・ゲームDP
・ゴールから逆算
・片方のプレイヤー側から見た場合の状態を考える。
・メモ化再帰でやってみる。
・ゲーム木の探索は普通のDPよりこっちのが分かりやすい気がする。
・ターン数は持ち歩くけど、スコアは持ち歩かないでメモに全部管理させる。
"""

import sys
sys.setrecursionlimit(10 ** 9)

A, B = map(int, input().split())
aA = list(map(int, input().split()))
bB = list(map(int, input().split()))

# メモ[左の山から取った枚数][右の山から取った枚数] = 先手番がこれ以降にとる価値
memo = [[-1] * (B+1) for i in range(A+1)]

def dfs(a, b):
    # 両方の山がなくなったらゲーム終了
    if a == A and b == B:
        return 0
    # 先手番かどうか
    if (a + b) % 2 == 0:
        # 既に見た局面ならメモの内容を返す
        if memo[a][b] != -1:
            return memo[a][b]
        mx = -1
        # 左の山にはある
        if a != A:
            mx = max(mx, dfs(a+1, b) + aA[a])
        # 右の山にはある
        if b != B:
            mx = max(mx, dfs(a, b+1) + bB[b])
        # ここより深い所を探し終わったので結果をメモする
        memo[a][b] = mx
        return mx
    # 後手番は得点が悪くなるように行動する
    # 先手番ではないので山a,bからの得点追加はない
    else:
        if memo[a][b] != -1:
            return memo[a][b]
        mn = float('inf')
        if a != A:
            mn = min(mn, dfs(a+1, b))
        if b != B:
            mn = min(mn, dfs(a, b+1))
        memo[a][b] = mn
        return mn

print(dfs(0, 0))
