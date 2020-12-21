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

X, Y = MAP()

if Y < abs(X*2):
    print(-1)
elif Y % 2 == 1:
    print(-1)
else:
    if X % 2 == 0 and Y % 4 == 0 \
            or X % 2 == 1 and Y % 4 == 2:
        print(Y//2)
    else:
        print(-1)
