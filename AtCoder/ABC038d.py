# -*- coding: utf-8 -*-

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
WH=[]
for i in range(N):
    w,h=MAP()
    WH.append((w, h))

WH.sort(key=itemgetter(1), reverse=True)
WH.sort(key=itemgetter(0), reverse=True)
cnt1=0
a=b=INF
for w, h in WH:
    if w<a and h<b:
        a=w
        b=h
        cnt1+=1

WH.sort(key=itemgetter(0), reverse=True)
WH.sort(key=itemgetter(1), reverse=True)
cnt2=0
a=b=INF
for w, h in WH:
    if w<a and h<b:
        a=w
        b=h
        cnt2+=1

print(max(cnt1, cnt2))
