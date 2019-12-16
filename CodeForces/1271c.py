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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, sx, sy = MAP()
XY = []
for i in range(N):
    x, y = MAP()
    XY.append((x, y))

a = b = c = d = 0
for x, y in XY:
    if x <= sx - 1:
        a += 1
    if x >= sx + 1:
        b += 1
    if y <= sy - 1:
        c += 1
    if y >= sy + 1:
        d += 1

mx = max(a, b, c, d)
print(mx)
if mx == a:
    print(sx-1, sy)
elif mx == b:
    print(sx+1, sy)
elif mx == c:
    print(sx, sy-1)
else:
    print(sx, sy+1)
