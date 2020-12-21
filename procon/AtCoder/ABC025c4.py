# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-9、自力AC！
・ゲームDP、メモ化再帰
・盤面のメモを一次元管理にして、効率良く進められたと思う。
"""

import sys
from collections import defaultdict

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

B = []
B.append(LIST())
B.append(LIST())
C = []
C.append(LIST())
C.append(LIST())
C.append(LIST())

total = sum(B[0]+B[1]+C[0]+C[1]+C[2])

def calc(state):
    res = 0
    for j in range(3):
        for i in range(2):
            if state[i*3+j] == state[(i+1)*3+j]:
                res += B[i][j]
    for i in range(3):
        for j in range(2):
            if state[i*3+j] == state[i*3+(j+1)]:
                res += C[i][j]
    return res

memo = defaultdict(lambda: -1)
def rec(turn, state):
    if memo[state] != -1:
        return memo[state]
    if turn == 9:
        res = calc(state)
        memo[state] = res
        return res
    state = list(state)
    if turn % 2 == 0:
        res = 0
        for i in range(9):
            if state[i] == '*':
                nxt = state.copy()
                nxt[i] = 'o'
                res = max(res, rec(turn+1, ''.join(nxt)))
    else:
        res = total
        for i in range(9):
            if state[i] == '*':
                nxt = state.copy()
                nxt[i] = 'x'
                res = min(res, rec(turn+1, ''.join(nxt)))
    memo[''.join(state)] = res
    return res

ans = rec(0, '*********')
print(ans)
print(total-ans)
