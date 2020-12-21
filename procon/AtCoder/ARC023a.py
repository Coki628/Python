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

def f(y, m, d):
    return 365*y + y//4 - y//100 + y//400 + (306*(m+1))//10 + d - 429

y = INT()
m = INT()
d = INT()

if m == 1 or m == 2:
    m += 12
    y -= 1

print(f(2014, 5, 17) - f(y, m, d))
