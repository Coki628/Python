# -*- coding: utf-8 -*-

"""
・numpyでループ1つ外した。結構速くなったぽいけどこれもTLE。
"""

import sys
import numpy as np
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
# Aが作れる組み合わせ全列挙
for i in range(1, N//2+1):
    for comb in combinations(A, i):
        team1 = sum(comb)
        team2 = Asm - team1
        if team1 == team2:
            continue
        # 強い方をteam1にする
        team1 = max(team1, team2)
        Ateams.add(team1)

Ateams = np.array(list(Ateams), dtype=np.int64)
win = np.zeros(len(Ateams), dtype=np.int64)
total = 0
for i in range(1, M):
    for comb in combinations(B, i):
        team1 = sum(comb)
        team2 = Bsm - team1
        if team1 <= team2:
            continue
        # team1,team2両方勝てるパターンの勝利数を+1する
        win = np.where((Ateams>team1)&(Asm-Ateams>team2), win+1, win)
        total += 1
res = win / total
print(res.max())
