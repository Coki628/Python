# -*- coding: utf-8 -*-

"""
・自力WA
・集まるのにちょうどいい点を探そうとしたが、その観点がそもそもミスっていた。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
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

N = INT()
X = Y = 0
XYC = []
for i in range(N):
    x, y, c = MAP()
    XYC.append((x, y, c))
    X += x
    Y += y

x1 = X / N
y1 = Y / N

mx = 0
for x, y, c in XYC:
    mx = max(mx, abs(x1-x), abs(y1-y))
print(mx)
