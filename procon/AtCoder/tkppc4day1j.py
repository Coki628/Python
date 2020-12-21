# -*- coding: utf-8 -*-

"""
・WA
・原因は、Aが取る最適をその都度変えてるから。
　あくまでAが取る組み合わせは1つ固めて、その各場合の勝率から
　最終的に最大を取るのが正解。
・あとAについてもsum-team1でteam2取れるから、team2持ってる必要なかった。。
・あとBの組み合わせは重複も含めて全パターン欲しいから、
　一応外ループMまで全部回した方がいい。
"""

import sys
from itertools import combinations
from bisect import bisect_right
from operator import itemgetter

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
        team1, team2 = max(team1, team2), min(team1, team2)
        Ateams.add((team1, team2))
# Aが作れるチームを、team1の昇順にしておく
Ateams = list(Ateams)
Ateams.sort(key=itemgetter(0))
# この後の二分探索用に行列入れ替え
Ateams = list(zip(*Ateams))

total = 0
win = 0
for i in range(1, M//2+1):
    for comb in combinations(B, i):
        team1 = sum(comb)
        team2 = Bsm - team1
        if team1 == team2:
            continue
        team1, team2 = max(team1, team2), min(team1, team2)
        # AがBのteam1にギリギリ勝てる位置
        idx = bisect_right(Ateams[0], team1)
        # その位置でteam2も勝てるかどうか
        if idx != len(Ateams[0]) and Ateams[1][idx] > team2:
            win += 1
        total += 1

print(win/total)
