# -*- coding: utf-8 -*-

import sys

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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    XY = []
    for i in range(N):
        x, y = MAP()
        XY.append((x+y, x, y))
    
    XY.sort()
    ax = ay = 0
    ans = []
    for d, x, y in XY:
        bx = x - ax
        by = y - ay
        if bx < 0 or by < 0:
            NO()
            break
        ans.append('R' * bx)
        ans.append('U' * by)
        ax += bx
        ay += by
    else:
        YES()
        print(''.join(ans))
