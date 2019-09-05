# -*- coding: utf-8 -*-

"""
・二分探索パートを外に出してみたけど、速度変わらず。
"""

import sys
from itertools import combinations
from bisect import bisect_left

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
for team1 in Ateam1s:
    team2 = Asm - team1
    # team1,2がそれぞれどこまで勝てるか
    team1win = bisect_left(Bteam1s, team1)
    team2win = bisect_left(Bteam2s, team2)
    # 両方の重複部分が、両方勝てる数
    bothwin = max(0, team1win+team2win-total)
    mx = max(mx, bothwin)
print(mx/total)
