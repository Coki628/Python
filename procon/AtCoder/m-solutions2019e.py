# -*- coding: utf-8 -*-

import sys
from math import log, exp

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

# 等差数列の和の公式を使う。
# [初項と末項を足した数]×[項数]/2
# →ダメだった。

x,d,n=MAP()
s=x
t=x+d*(n-1)
logs=log(s)
logt=log(t)

logans=(logs+logt)*n/2

print(exp(logans))
