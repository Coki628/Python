# -*- coding: utf-8 -*-

"""
普通にDFS　→盛大にTLE
"""
 
import sys
 
# 再帰呼び出しの回数制限(デフォルト1000)
sys.setrecursionlimit(10 ** 9)
 
N, H = map(int, input().split())
A, B, C, D, E = map(int, input().split())
 
total = float('inf')
def dfs(money, hp, day):
    # 規定の日数に達したら、金額を確認する
    global total
    if day == N:
        total = min(money, total)
        return
 
    if hp <= E:
        # 普通の食事をする
        dfs(money + A, hp + B, day + 1)
        # 質素な食事をする
        dfs(money + C, hp + D, day + 1)
    else:
        # 食べない
        dfs(money, hp - E, day + 1)
 
dfs(0, H, 0)
print(total)
