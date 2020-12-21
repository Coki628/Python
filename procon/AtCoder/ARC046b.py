# -*- coding: utf-8 -*-

"""
・自力AC！
・ゲーム木の探索、ミニマックス法
・規則性を見つけて、その通りに出力
　(結構見つけるの大変だったけど)
・公式解は考察だけで解いてたけど、なかなか小さいケース試すだけで導き出すの大変だよなぁ。
"""

import sys
from math import sqrt

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

N = INT()
A, B = MAP()

memo = list4d(2, 50, 50, 50, -1)
def rec(turn, n):
    if memo[turn][n][A][B] != -1:
        return memo[turn][n][A][B]
    if n == 0:
        return turn
    if turn%2 == 0:
        res = 0
        for i in range(1, A+1):
            res |= rec(1-turn, n-i)
            if n == i:
                break
    else:
        res = 1
        for i in range(1, B+1):
            res &= rec(1-turn, n-i)
            if n == i:
                break
    memo[turn][n][A][B] = res
    return res

def divisor_set(N: int) -> set:
    """ 約数の列挙・個数 """
    # 1とその数はデフォで入れとく
    s = {1, N}
    # 終点はルート切り捨て
    for i in range(2, int(sqrt(N))+1):
        # 割り切れるなら、iとN/iを追加
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

# 調査用
# aloses = []
# for i in range(1, 41):
#     print('N: {0}'.format(i))
#     alose = []
#     for A in range(1, i+1):
#         li = []
#         for B in range(1, i+1):
#             res = rec(0, i)
#             # print('N: {0}, A: {1}, B: {2}, res: {3}'.format(i, A, B, res))
#             li.append(res)
#             if A == B and res == 0:
#                 alose.append(A)
#             if A > B and res == 0:
#                 print('N: {0}, A: {1}, B: {2}, res: {3}'.format(i, A, B, res))
#             if A < B and res == 1:
#                 print('N: {0}, A: {1}, B: {2}, res: {3}'.format(i, A, B, res))
#         # print(*li)
#     aloses.append(alose)
# for i, alose in enumerate(aloses):
#     print('N: {0}'.format(i+1))
#     print(*alose)
#     divs = divisor_set(i+1)
#     divs = sorted(set(div-1 for div in divs))
#     divs.remove(0)
#     print(*divs)

if A > B or A >= N:
    print('Takahashi')
elif A < B:
    print('Aoki')
# AとBの石を取れる数が同じ時
elif A == B:
    divs = divisor_set(N)
    divs = set(div-1 for div in divs)
    # 取れる数が、Nの約数-1に該当しなければAが勝つ
    if A not in divs:
        print('Takahashi')
    else:
        print('Aoki')
