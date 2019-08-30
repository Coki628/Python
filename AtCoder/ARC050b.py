# -*- coding: utf-8 -*-

"""
・しょうもない場合分けしか生えなかった。サンプルでWA。。
"""

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

R, B = MAP()
x, y = MAP()

ans1 = 0
r = R // x
r2 = R % x
if B >= r:
    b = B - r
    ans1 += r
    b2 = b // y
    ans1 += min(r2, b2)
else:
    ans1 += B

ans2 = 0
b = B // y
b2 = B % y
if R >= b:
    r = R - b
    ans2 += b
    r2 = r // x
    ans2 += min(r2, b2)
else:
    ans2 += R

print(max(ans1, ans2))
