# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/tkppc4-1/submissions/6555137
・で、これで勝率は正しくなったけどループ1重増えてTLE
"""

import sys
from itertools import combinations

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
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

N, M = MAP()
A = LIST()
B = LIST()

Asm = sum(A)
Bsm = sum(B)

Ateams = set()
# 組み合わせ全列挙
for i in range(1, N//2+1):
    for comb in combinations(A, i):
        team1 = sum(comb)
        team2 = Asm - team1
        if team1 == team2:
            continue
        # 強い方をteam1にする
        team1 = max(team1, team2)
        Ateams.add(team1)

mx = 0
for Ateam1 in Ateams:
    total = 0
    win = 0
    for i in range(1, M):
        for comb in combinations(B, i):
            team1 = sum(comb)
            team2 = Bsm - team1
            if team1 <= team2:
                continue
            if Ateam1 > team1 and Asm-Ateam1 > team2:
                win += 1
            total += 1
    mx = max(mx, win/total)
print(mx)
