# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-9、自力AC！
・ゲームDP、メモ化再帰
・メモが1000^2=100万あるけど、各回の遷移が2回だけなので問題ない。
　これが内側でNのループ回してたりするとダメ。(ABC078d6)
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
INF = float('inf')
MOD = 10 ** 9 + 7

N, M = MAP()
A = LIST()
B = LIST()

memo = list2d(N+1, M+1, -1)
def rec(n, m):
    if memo[n][m] != -1:
        return memo[n][m]
    if n == N and m == M:
        return 0
    # 山札の合計からターンは分かる
    turn = (n+m) % 2
    # 先手番
    if turn % 2 == 0:
        res = -1
        if n != N:
            res = max(res, rec(n+1, m)+A[n])
        if m != M:
            res = max(res, rec(n, m+1)+B[m])
    # 後手番
    else:
        res = INF
        if n != N:
            res = min(res, rec(n+1, m))
        if m != M:
            res = min(res, rec(n, m+1))
    memo[n][m] = res
    return res

print(rec(0, 0))
