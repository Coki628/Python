# -*- coding: utf-8 -*-

"""
・400点自力AC
・素直に貪欲でOKか迷うとこだったけど、最初WAしたのは際の処理ミスってただけだった。
"""

import sys
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

N=INT()
AB=[]
for i in range(N):
    a,b=MAP()
    AB.append((a, b))

AB.sort(key=itemgetter(1))
time=0
for i in range(N):
    a,b=AB[i]
    time+=a
    if b<time:
        No()
        exit()
Yes()
