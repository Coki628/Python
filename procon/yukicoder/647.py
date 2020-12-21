# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・計算量がN*M<=1000万だけど、制限4.5秒ある。pypyじゃなくても大丈夫だった。pythonAC3.1秒。
"""

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

N = INT()
AB = []
for i in range(N):
    a, b = MAP()
    AB.append((a, b))
M = INT()
XY = []
for i in range(M):
    x, y = MAP()
    XY.append((x, y))

C = [0] * M
for a, b in AB:
    for i, (x, y) in enumerate(XY):
        if x <= a and y >= b:
            C[i] += 1
mx = max(C)
ans = []
if mx == 0:
    ans.append(0)
else:
    for i, c in enumerate(C):
        if c == mx:
            ans.append(i+1)
for a in ans:
    print(a)
