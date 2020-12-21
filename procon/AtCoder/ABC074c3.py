# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC！
・集合管理するDP
・公式解と違うけど、きっちり考えて通せてるからまあいいでしょう。
・計算量は3000^2=900万、pypyAC0.3秒、pythonAC2.2秒。(制約3秒)
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

A, B, C, D, E, F =  MAP()
A *= 100
B *= 100
# 溶ける最大濃度
limit = E / (100+E)

# dp[i][j] := 水i、砂糖jの状態を実現可能か
dp = list2d(F+1, F+1, 0)
dp[0][0] = 1
for i in range(F+1):
    for j in range(F+1):
        # 枝刈り
        if i+j > F:
            break
        # 実現可能な状態なら、4つの行動を遷移させる
        if dp[i][j]:
            if i+A <= F:
                dp[i+A][j] = 1
            if i+B <= F:
                dp[i+B][j] = 1
            if j+C <= F:
                dp[i][j+C] = 1
            if j+D <= F:
                dp[i][j+D] = 1

ans1 = ans2 = 0
mx = -1
for a in range(F+1):
    for b in range(F+1):
        if a == b == 0:
            continue
        if a+b > F:
            break
        if dp[a][b]:
            x = b / (a+b)
            # 濃度が濃すぎて溶け切らない
            if x > limit:
                continue
            # 最大濃度の更新
            if mx < x:
                mx = x
                ans1 = a + b
                ans2 = b
print(ans1, ans2)
