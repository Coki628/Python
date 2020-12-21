# -*- coding: utf-8 -*-

"""
全探索
・再帰で作る
・2^20でTLEするからメモ化した
"""

N = int(input())
aN = list(map(int, input().split()))
Q = int(input())
mQ = list(map(int, input().split()))

def dfs(cur, depth, ans):
    # 終了条件
    if cur == ans:
        return True
    # memo[現在位置]に数値curがあれば、そこから先はやっても同じだからやらない
    if cur in memo[depth]:
        return False
    memo[depth].add(cur)
    # 全探索
    for i in range(depth, N):
        if dfs(cur+aN[i], i+1, ans):
            return True
    # 見つからなかった
    return False

for i in range(Q):
    memo = [set() for j in range(N+1)]
    if dfs(0, 0, mQ[i]):
        print('yes')
    else:
        print('no')
