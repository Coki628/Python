# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・日付管理。実務で使いそうな感じ。
"""

import sys
from datetime import date, timedelta

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
INF = float('inf')
MOD = 10 ** 9 + 7

y, m, d = map(int, input().split('/'))

day = date(y, m, d)
while y % (m*d) != 0:
    day = day + timedelta(days=1)
    y, m, d = day.year, day.month, day.day

print(day.strftime('%Y/%m/%d'))
