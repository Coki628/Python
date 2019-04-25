# -*- coding: utf-8 -*-

import sys

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
XY=[[None] for i in range(N)]
x,y=MAP()
XY[0]=(abs(x)+abs(y), x, y)
od_ev=(abs(x)+abs(y))%2
for i in range(1, N):
    x,y=MAP()
    # まず大前提として偶奇が合うかどうか
    if od_ev!=(abs(y)+abs(x))%2:
        print(-1)
        exit()
    XY[i]=(abs(x)+abs(y), x, y)

XY.sort()
arms={0}
for i in range(N):
    sm,x,y=XY[i]
    arms.add(min(abs(x), abs(y)))
    arms.add(sm)
arms=sorted(arms)
d=[]
for i in range(1, len(arms)):
    d.append(arms[i]-arms[i-1])

if len(d)>40:
    print(-1)
    exit()

print()
