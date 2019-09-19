# -*- coding: utf-8 -*-

"""
・400点自力AC！
・ゲームDP、メモ化再帰
・ゲーム木の探索、ミニマックス法
・自ターンでは1つでも1が返れば1、相手ターンでは1つでも0が返れば0を返す。
・制約上そのまま使っても通せないので、規則性を見つけてその通りに出力。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

A, B = MAP()

memo = list4d(2, 100, 100, 2, -1)
def rec(turn, a, b, board):
    if memo[turn][a][b][board] != -1:
        return memo[turn][a][b][board]
    if a==0 and b==0:
        memo[turn][a][b][board] = board
        return board
    if turn%2 == 0:
        res = 0
        if a != 0:
            res |= rec(1-turn, a-1, b, board)
        if b != 0:
            res |= rec(1-turn, a, b-1, 1-board)
    else:
        res = 1
        if a != 0:
            res &= rec(1-turn, a-1, b, board)
        if b != 0:
            res &= rec(1-turn, a, b-1, board)
    memo[turn][a][b][board] = res
    return res

# table = list2d(20, 20, 0)
# for i in range(20):
#     for j in range(20):
#         res = rec(0, i, j, 0)
#         table[i][j] = res
#         print('{0},{1}: {2}'.format(i, j, res))
# for i in range(20):
#     print(*table[i])

if A%2 == 0:
    if B%4 in [1,2]:
        print('Angel')
    else:
        print('Devil')
else:
    if B%4 in [1,2,3]:
        print('Angel')
    else:
        print('Devil')
