# -*- coding: utf-8 -*-

"""
・なんとか自力AC
・再帰探索、複雑な構造の生成
・再帰の方針は最初から立ってたけど、条件を整えるのに時間がかかった。。
・計算量はNが10なのでだいたい10!=363万くらい。pythonAC1.3秒、pypyAC0.8秒。
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

ans = set()
def rec(i, S):
    if S in ans:
        return
    if i == N:
        ans.add(S)
        return
    # 今までに出現した文字は全部追加できる
    for c in S:
        rec(i+1, S+c)
    # 今までに出現した中で最大の文字の次も追加できる
    rec(i+1, S+chr(ord(max(S))+1))
rec(1, 'a')
ans = sorted(ans)
[print(s) for s in ans]
