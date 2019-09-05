# -*- coding: utf-8 -*-

"""
・二分探索パートをnumpyでまとめて処理
・これでpythonもAC、2.8秒でギリギリだけど。
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

# Aの組み合わせ全列挙
Ateam1s = set()
for i in range(1, N//2+1):
    for comb in combinations(A, i):
        team1 = sum(comb)
        team2 = Asm - team1
        if team1 == team2:
            continue
        # 強い方をteam1にする
        team1 = max(team1, team2)
        Ateam1s.add(team1)

# Bの組み合わせ全列挙
Bteam1s, Bteam2s = [], []
for i in range(1, M):
    for comb in combinations(B, i):
        team1 = sum(comb)
        team2 = Bsm - team1
        if team1 <= team2:
            continue
        Bteam1s.append(team1)
        Bteam2s.append(team2)
Bteam1s.sort()
Bteam2s.sort()

mx = 0
total = len(Bteam1s)
Ateam1s = np.array(list(Ateam1s), dtype=np.int64)
Ateam2s = Asm - Ateam1s
# team1,2がそれぞれどこまで勝てるか
team1wins = np.searchsorted(Bteam1s, Ateam1s)
team2wins = np.searchsorted(Bteam2s, Ateam2s)
# 両方の重複部分が、両方勝てる数
bothwins = np.maximum(0, team1wins+team2wins-total)
mx = bothwins.max()
print(mx/total)
