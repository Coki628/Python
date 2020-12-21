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

H,W,N=MAP()

edges=set()
for i in range(N):
    r,c=MAP()
    if r>1:
        edges.add(((r-1, c), (r, c)))
    if c>1:
        edges.add(((r, c-1), (r, c)))
    if r<H:
        edges.add(((r, c), (r+1, c)))
    if c<W:
        edges.add(((r, c), (r, c+1)))

ans=(H-1)*W+(W-1)*H
print(ans-len(edges))
