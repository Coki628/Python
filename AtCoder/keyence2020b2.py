# -*- coding: utf-8 -*-

"""
・なんとか自力AC。これ200点じゃないだろ。。
・最初、座圧していもすとか迷走しまくった。。
・結局のところ、基本の区間スケジューリング。
　各区間を右端でソートして、終わり早いやつから貪欲に取る。
・両端とも境界を含まない閉区間で、両側-1,+1したらハマった。
　それだと必要以上に隙間が空くと分かって、0.5ずつにしてAC。
・公式解説見たけど、両端閉区間てことはその座標にどう重なってもOKなんだから、
　特に増減させずに等号付きで比較すればそれで良かった。。
"""

import sys
from operator import itemgetter

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

N = INT()
LR = []
for i in range(N):
    x, l = MAP()
    LR.append((x-l, x+l))

LR.sort(key=itemgetter(1))
cur = -INF
cnt = 0
for l, r in LR:
    if cur <= l:
        cur = r
        cnt += 1
print(cnt)
