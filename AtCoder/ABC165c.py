# -*- coding: utf-8 -*-

"""
・コンテスト中は計算量自信なかったし再帰だと思ったからC++で通したけどこっちでもやっとく。
・重複組み合わせの全列挙
・combinations_with_replacementは存在は知ってたけど、実際決め手になったのは初めてかも。
・計算量はnHr(M, N)=nCr(M+N-1, N)=nCr(19, 10)=9万ちょっと、に内側の処理でQ<=50だから、450万くらい。
　pythonAC0.9秒。やっぱ新環境で速くなってんのかな。数百万がpypyじゃなくてもよく通る。
"""

import sys
from itertools import combinations_with_replacement

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

N, M, Q = MAP()
ABCD = []
for i in range(Q):
    a, b, c, d = MAP()
    ABCD.append((a-1, b-1, c, d))

ans = 0
for comb in combinations_with_replacement(range(1, M+1), N):
    cnt = 0
    for a, b, c, d in ABCD:
        if comb[b] - comb[a] == c:
            cnt += d
    ans = max(ans, cnt)
print(ans)
