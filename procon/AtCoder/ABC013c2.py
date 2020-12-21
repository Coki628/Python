# -*- coding: utf-8 -*-

"""
メモ化再帰で高速化
(ここまでは自力)
"""

import sys
from collections import defaultdict

# 再帰呼び出しの回数制限(デフォルト1000)
sys.setrecursionlimit(10 ** 9)

N, H = map(int, input().split())
A, B, C, D, E = map(int, input().split())

# メモ化再帰({ある日、ある満腹度の時の金額}, 
#           {ある日、ある金額の時の満腹度})
memo = [[defaultdict(lambda: float('inf')), 
    defaultdict(lambda: -1)] for i in range(N)]

total = float('inf')
def dfs(money, hp, day):
    # 規定の日数に達したら、金額を確認する
    global total
    if day == N:
        total = min(money, total)
        return
    # 訪問済や値の悪いのはなるべくやらないように
    if (memo[day-1][0][hp] <= money
            or memo[day-1][1][money] >= hp):
        return
    memo[day-1][0][hp] = money
    memo[day-1][1][money] = hp

    if hp <= E:
        # 質素な食事をする
        dfs(money + C, hp + D, day + 1)
        # 普通の食事をする
        dfs(money + A, hp + B, day + 1)
    else:
        # 食べない
        dfs(money, hp - E, day + 1)

dfs(0, H, 0)
print(total)
