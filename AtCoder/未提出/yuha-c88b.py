# -*- coding: utf-8 -*-

"""
・ゲームDP、メモ化再帰
・負けを押し付けられるかどうか。
・ゲームDP書いてしまったけど、これは手元で小さいの書いて確認するだけのが早かった。。
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

N = INT()

# memo[i] := 残りコイン数i枚で勝てるか
memo = [-1] * (N+1)
def rec(cnt):
    if cnt <= 0:
        return False
    if memo[cnt] != -1:
        return memo[cnt]
    for i in range(1, 4):
        if not rec(cnt-i):
            memo[cnt] = True
            return True
    memo[cnt] = False
    return False

if N % 4 == 0:
    print('GO')
else:
    print('SEN')
